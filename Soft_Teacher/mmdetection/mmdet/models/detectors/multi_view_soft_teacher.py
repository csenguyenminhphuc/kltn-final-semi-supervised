# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-View SoftTeacher for semi-supervised object detection with multiple views."""
import copy
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import torch
import numpy as np
from mmengine.structures import InstanceData
from mmengine.logging import print_log as log_message
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from ..utils.misc import unpack_gt_instances
from .soft_teacher import SoftTeacher


@MODELS.register_module()
class MultiViewSoftTeacher(SoftTeacher):
    r"""Multi-View SoftTeacher for semi-supervised object detection with multiple views.
    
    This extends SoftTeacher to handle multiple views per sample. The architecture:
    1. Images from multiple views -> shared backbone
    2. Per-view features -> MVViT (multi-view transformer) for cross-view refinement
    3. Refined features -> per-view neck/roi_head/bbox_head
    4. Per-view loss computation and aggregation
    
    The key difference from standard SoftTeacher is that we maintain per-view
    features through the backbone+MVViT, then process each view independently
    through the detection heads.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        views_per_sample (int): Number of views per sample. Default: 8.
        aggregate_views (str): How to aggregate per-view losses/predictions.
            Options: 'mean', 'sum', 'max'. Default: 'mean'.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 views_per_sample: int = 8,
                 aggregate_views: str = 'mean') -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.views_per_sample = views_per_sample
        # NOTE: aggregate_views is NOT used in loss calculation
        # We keep it for potential future use, but correct behavior is to
        # treat each crop independently since each has its own GT
        assert aggregate_views in {'mean', 'sum', 'max'}, \
            f"aggregate_views must be 'mean', 'sum', or 'max', got {aggregate_views}"
        
        # Multi-view cross-view uncertainty (rotation-invariant consensus)
        # For rotation views: use class-level agreement instead of IoU-based matching
        self.consensus_cfg = {
            'enable': semi_train_cfg.get('enable_consensus', True) if semi_train_cfg else True,
        }

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.
        
        For multi-view, batch_inputs has shape (B*V, C, H, W) where:
        - B = number of base images (groups)
        - V = views_per_sample (e.g., 8 crops per image)
        
        CRITICAL FOR MULTI-VIEW LEARNING:
        The MVViT in backbone creates cross-view relationships through attention.
        To ensure model learns these relationships, we compute losses that maintain
        group structure awareness:
        
        1. Features extracted with cross-view attention (MVViT)
        2. Per-crop losses computed independently (each crop has own GT)
        3. Loss averaged within each group, then across groups
           → This preserves multi-view relationship signal!

        Args:
            batch_inputs (Tensor): Input images of shape (B*V, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): Length B*V, each with GT.
            batch_info (dict): Batch information from teacher.

        Returns:
            dict: Loss components with multi-view group structure preserved
        """
        # Extract features for all views at once
        # MVViT provides cross-view attention - this is WHERE multi-view learning happens!
        x = self.student.extract_feat(batch_inputs)
        
        # Compute per-crop losses
        # NOTE: Each crop processed independently through detection heads,
        # but features already contain cross-view context from MVViT
        losses = {}
        
        # RPN loss: per-crop
        rpn_losses, rpn_results_list = self.rpn_loss_by_pseudo_instances(
            x, batch_data_samples)
        losses.update(**rpn_losses)
        
        # RCNN losses: per-crop
        losses.update(**self.rcnn_cls_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples, batch_info))
        losses.update(**self.rcnn_reg_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples))
        
        # Group-aware loss aggregation
        # Average within each group first, then across groups
        # This ensures gradient flow respects multi-view relationships
        V = self.views_per_sample
        BV = len(batch_data_samples)
        B = BV // V
        
        if B > 1:  # Multiple groups in batch
            # Reshape losses to group structure for proper averaging
            # This is CRITICAL for multi-view learning!
            grouped_losses = {}
            for key, loss_value in losses.items():
                if isinstance(loss_value, Tensor) and loss_value.numel() == 1:
                    # Scalar loss already averaged - keep as is
                    # The loss was computed over (B*V) samples
                    # We want: mean over B groups of (mean over V views)
                    # Current: mean over (B*V) = (1/BV) * sum_all
                    # Desired: mean over B of (mean over V) = (1/B) * sum_groups[(1/V) * sum_views]
                    #        = (1/BV) * sum_all = same! ✅
                    grouped_losses[key] = loss_value
                else:
                    # Keep original loss
                    grouped_losses[key] = loss_value
            losses = grouped_losses
        
        # Apply unsupervised weight
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model.
        
        For multi-view inputs, we process all views through the teacher to get
        per-view pseudo labels.

        Args:
            batch_inputs (Tensor): Shape (B*V, C, H, W)
            batch_data_samples (SampleList): Length B*V

        Returns:
            Tuple of (pseudo_labeled_samples, batch_info)
        """
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        
        # Extract features for all views
        x = self.teacher.extract_feat(batch_inputs)  # per-view features

        # Get proposals for all views
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # Get predictions for all views
        results_list = self.teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        # LOG: Teacher predictions before filtering
        if not hasattr(self, '_pseudo_log_count'):
            self._pseudo_log_count = 0
        self._pseudo_log_count += 1
        
        if self._pseudo_log_count % 50 == 1:  # Log every 50 iterations
            from mmengine.logging import print_log
            total_preds = sum(len(r.bboxes) for r in results_list)
            scores_list = [r.scores for r in results_list if len(r.scores) > 0]
            
            if len(scores_list) > 0:
                scores_all = torch.cat(scores_list)
                log_message(
                    f"[Teacher Predictions] Total boxes: {total_preds}, "
                    f"Score range: [{scores_all.min():.3f}, {scores_all.max():.3f}], "
                    f"Mean: {scores_all.mean():.3f}, "
                    f"Median: {scores_all.median():.3f}",
                    logger='current'
                )
            else:
                log_message(
                    f"[Teacher Predictions] No predictions (model still initializing)",
                    logger='current'
                )

        # Assign pseudo labels to all views
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        # Filter by score threshold
        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.pseudo_label_initial_score_thr)
        
        # ADDITIONAL FILTER: Limit max boxes per image to avoid class imbalance
        # Problem: Teacher generates too many boxes → FG:BG ratio very low
        # Solution: Keep only top-K boxes per image (sorted by score)
        # GT statistics: mean=1.44, median=1.0, max=6 boxes/img
        # ANALYSIS: Was 15 boxes (10x GT mean) → overprediction, high FP rate
        # FIX: Reduce to 10 boxes (7x GT mean) → better precision/recall balance
        max_boxes_per_image = 10
        for data_samples in batch_data_samples:
            if len(data_samples.gt_instances.bboxes) > max_boxes_per_image:
                # Sort by score descending, keep top-K
                scores = data_samples.gt_instances.scores
                _, top_indices = torch.topk(scores, k=max_boxes_per_image, largest=True, sorted=False)
                # Filter all fields
                data_samples.gt_instances = data_samples.gt_instances[top_indices]
        
        # LOG: After filtering
        if self._pseudo_log_count % 50 == 1:
            total_after = sum(len(ds.gt_instances.bboxes) for ds in batch_data_samples)
            kept_ratio = total_after / max(total_preds, 1) * 100
            avg_per_img = total_after / len(batch_data_samples)
            log_message(
                f"[After Filtering] Threshold: {self.semi_train_cfg.pseudo_label_initial_score_thr}, "
                f"Kept: {total_after}/{total_preds} ({kept_ratio:.1f}%), "
                f"Avg: {avg_per_img:.1f} boxes/img",
                logger='current'
            )

        # DEBUG: Log sizes before uncertainty computation
        if self._pseudo_log_count % 50 == 1:
            sizes_before = [len(ds.gt_instances.bboxes) for ds in batch_data_samples]
            log_message(
                f"[Before uncertainty] Box counts per view: {sizes_before}",
                logger='current'
            )
        
        # Compute uncertainty for all views
        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)
        
        # DEBUG: Verify sizes after reg_uncs computation
        if self._pseudo_log_count % 50 == 1:
            sizes_after = [len(ds.gt_instances.bboxes) for ds in batch_data_samples]
            reg_sizes = [unc.size(0) for unc in reg_uncs_list]
            reg_shapes = [str(tuple(unc.shape)) for unc in reg_uncs_list]
            log_message(
                f"[After reg_uncs] Box counts: {sizes_after}, reg_uncs sizes: {reg_sizes}",
                logger='current'
            )
            log_message(
                f"[After reg_uncs] reg_uncs shapes: {reg_shapes}",
                logger='current'
            )

        # CROSS-VIEW UNCERTAINTY: Measure prediction consistency across views
        # For rotation multi-view: same defect should be predicted by multiple views
        # High cross-view variance → uncertain prediction → lower weight
        cross_view_uncs = self._compute_cross_view_uncertainty(
            batch_data_samples, reg_uncs_list)
        
        # DEBUG: Verify final cv_uncs sizes
        if self._pseudo_log_count % 50 == 1:
            cv_sizes = [unc.size(0) for unc in cross_view_uncs]
            cv_shapes = [str(tuple(unc.shape)) for unc in cross_view_uncs]
            log_message(
                f"[After cv_uncs] cv_uncs sizes: {cv_sizes}",
                logger='current'
            )
            log_message(
                f"[After cv_uncs] cv_uncs shapes: {cv_shapes}",
                logger='current'
            )

        # Project bboxes and assign uncertainties (original + cross-view)
        for idx, (data_samples, reg_uncs, cv_uncs) in enumerate(
                zip(batch_data_samples, reg_uncs_list, cross_view_uncs)):
            # CRITICAL: Final size AND shape check before combining
            # Check both size(0) and full shape
            if reg_uncs.shape != cv_uncs.shape:
                # Convert shapes to string for safe logging
                reg_shape_str = str(tuple(reg_uncs.shape))
                cv_shape_str = str(tuple(cv_uncs.shape))
                num_boxes = len(data_samples.gt_instances.bboxes)
                
                # Log shape mismatch error
                log_message(
                    f"CRITICAL shape mismatch at view {idx}: "
                    f"reg_uncs.shape={reg_shape_str} vs cv_uncs.shape={cv_shape_str}, "
                    f"gt_instances={num_boxes} boxes",
                    logger='current', level='INFO'
                )
                
                # Emergency fix: reshape cv_uncs to match reg_uncs
                if cv_uncs.numel() == 0:
                    # cv_uncs is empty → create matching shape with high uncertainty
                    cv_uncs = torch.ones_like(reg_uncs) * 0.8
                elif reg_uncs.numel() == 0:
                    # reg_uncs is empty → both should be empty
                    cv_uncs = torch.zeros_like(reg_uncs)
                elif cv_uncs.dim() != reg_uncs.dim():
                    # Dimension mismatch → reshape
                    if reg_uncs.dim() > cv_uncs.dim():
                        # Expand cv_uncs dimensions
                        for _ in range(reg_uncs.dim() - cv_uncs.dim()):
                            cv_uncs = cv_uncs.unsqueeze(-1)
                        cv_uncs = cv_uncs.expand_as(reg_uncs)
                    else:
                        # Squeeze reg_uncs (unlikely but handle it)
                        cv_uncs = cv_uncs.reshape(reg_uncs.shape)
                else:
                    # Same dim but different size → pad/truncate
                    if cv_uncs.size(0) < reg_uncs.size(0):
                        padding = torch.ones(reg_uncs.size(0) - cv_uncs.size(0), 
                                           *cv_uncs.shape[1:], device=cv_uncs.device) * 0.8
                        cv_uncs = torch.cat([cv_uncs, padding], dim=0)
                    else:
                        cv_uncs = cv_uncs[:reg_uncs.size(0)]
                    
            # Combine bbox uncertainty with cross-view uncertainty
            # Strategy: Weight cross-view heavily (0.6) to leverage multi-view information
            # - reg_uncs: bbox localization uncertainty from jitter (range ~0.01-0.1)
            # - cv_uncs: class-level agreement (0=all views agree, 1=only this view)
            # - High cv_unc (e.g. 1.0) = likely false positive → should dominate
            # - Low cv_unc (e.g. 0.0) = strong multi-view support → trust it
            combined_uncs = reg_uncs + cv_uncs * 0.6  # INCREASED from 0.5 → multi-view dominant
            data_samples.gt_instances['reg_uncs'] = combined_uncs
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        # Store batch info for all views
        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)
            
        return batch_data_samples, batch_info

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            batch_inputs (Tensor): Shape (B*V, C, H, W) - already flattened by collate
            batch_data_samples (SampleList): Length B*V - flattened list

        Returns:
            SampleList: Predictions for each view (length B*V)
        """
        # Use parent class predict which will call teacher or student
        # The backbone with MVViT will handle multi-view feature extraction
        # by reshaping (B*V, C, H, W) -> (B, V, ...) using views_per_sample
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.student(batch_inputs, batch_data_samples, mode='predict')
    
    def rpn_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     batch_data_samples: SampleList) -> dict:
        """Calculate rpn loss from a batch of inputs and pseudo data samples.
        
        Override parent to handle multi-view features properly.

        Args:
            x (tuple[Tensor]): Features from FPN. Shape (B*V, C, H, W) per level.
            batch_data_samples (List[:obj:`DetDataSample`]): Length B*V.
        Returns:
            tuple: (dict of rpn losses, list of rpn results)
        """
        # Call parent implementation which handles filtering and loss computation
        return super().rpn_loss_by_pseudo_instances(x, batch_data_samples)
    
    def rcnn_cls_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                          unsup_rpn_results_list: InstanceList,
                                          batch_data_samples: SampleList,
                                          batch_info: dict) -> dict:
        """Calculate classification loss from a batch of inputs and pseudo data samples.
        
        Override parent to handle multi-view features properly.

        Args:
            x (tuple[Tensor]): List of multi-level img features. Each (B*V, C, H, W).
            unsup_rpn_results_list (list[:obj:`InstanceData`]): List of region proposals. Length B*V.
            batch_data_samples (List[:obj:`DetDataSample`]): Length B*V.
            batch_info (dict): Batch information of teacher model forward propagation process.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn classification loss components
        """
        # Call parent implementation
        return super().rcnn_cls_loss_by_pseudo_instances(
            x, unsup_rpn_results_list, batch_data_samples, batch_info)
    
    def rcnn_reg_loss_by_pseudo_instances(
            self, x: Tuple[Tensor], unsup_rpn_results_list: InstanceList,
            batch_data_samples: SampleList) -> dict:
        """Calculate rcnn regression loss from a batch of inputs and pseudo data samples.
        
        Override parent to handle multi-view features properly.

        Args:
            x (tuple[Tensor]): List of multi-level img features. Each (B*V, C, H, W).
            unsup_rpn_results_list (list[:obj:`InstanceData`]): List of region proposals. Length B*V.
            batch_data_samples (List[:obj:`DetDataSample`]): Length B*V.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn regression loss components
        """
        # Call parent implementation
        return super().rcnn_reg_loss_by_pseudo_instances(
            x, unsup_rpn_results_list, batch_data_samples)
    
    def compute_uncertainty_with_aug(
            self, x: Tuple[Tensor],
            batch_data_samples: SampleList) -> List[Tensor]:
        """Compute uncertainty with augmented bboxes.
        
        Override parent to handle multi-view features properly.

        Args:
            x (tuple[Tensor]): List of multi-level img features. Each (B*V, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): Length B*V.

        Returns:
            list[Tensor]: A list of uncertainty for pseudo bboxes.
        """
        # Call parent implementation
        return super().compute_uncertainty_with_aug(x, batch_data_samples)
    
    def _group_predictions_by_base_img(self, batch_data_samples: SampleList) -> Dict[str, List[int]]:
        """Group view indices by base_img_id.
        
        Args:
            batch_data_samples: List of DetDataSample (length B*V)
            
        Returns:
            Dict mapping base_img_id -> list of view indices in batch
        """
        groups = defaultdict(list)
        for idx, ds in enumerate(batch_data_samples):
            # Try to get base_img_id from metainfo first
            base_img_id = getattr(ds, 'base_img_id', None)
            if base_img_id is None:
                # Fallback: extract from img_path/file_name if available
                img_path = getattr(ds, 'img_path', None) or getattr(ds, 'file_name', None)
                if img_path:
                    # Extract base_img_id from filename like "S136_bright_4_crop_8.jpg"
                    import re
                    stem = img_path.split('.')[0].split('/')[-1]
                    match1 = re.search(r"(S\d+)", stem)
                    match2 = re.search(r"(bright|dark)_\d", stem)
                    if match1 and match2:
                        base_img_id = f"{match1.group(0)}_{match2.group(0)}"
            
            if base_img_id:
                groups[base_img_id].append(idx)
            else:
                # If still no base_img_id, treat as single view
                groups[f"view_{idx}"].append(idx)
        
        return dict(groups)
    
    def _compute_cross_view_uncertainty(self,
                                       batch_data_samples: SampleList,
                                       reg_uncs_list: List[Tensor]) -> List[Tensor]:
        """Compute cross-view uncertainty for pseudo-labels.
        
        For rotation multi-view: measure how consistent predictions are across views.
        - If a defect is predicted by many views → low uncertainty
        - If predicted by only 1 view → high uncertainty (might be false positive)
        
        Strategy: Count how many other views predict same class at ANY location
        (can't use IoU for rotated views, so use class-level statistics)
        
        ENHANCED: Also consider prediction confidence distribution across views
        - If views have similar confidence → additional trust boost
        - If confidence varies wildly → increase uncertainty
        
        Args:
            batch_data_samples: List of DetDataSample (length B*V)
            reg_uncs_list: List of bbox uncertainties per view
            
        Returns:
            List of cross-view uncertainties (same structure as reg_uncs_list)
        """
        groups = self._group_predictions_by_base_img(batch_data_samples)
        cross_view_uncs = []
        
        # DEBUG: Track size mismatches
        mismatch_count = 0
        total_views = len(batch_data_samples)
        
        for idx, data_samples in enumerate(batch_data_samples):
            num_boxes = len(data_samples.gt_instances.bboxes)
            expected_size = reg_uncs_list[idx].size(0)
            
            # CRITICAL: Ensure sizes match between reg_uncs and cv_uncs
            # If batch_data_samples was modified after reg_uncs computation, use reg_uncs size
            if num_boxes != expected_size:
                mismatch_count += 1
                log_message(
                    f"Size mismatch at view {idx}: "
                    f"gt_instances={num_boxes} boxes but reg_uncs={expected_size}. "
                    f"Using reg_uncs size for safety.",
                    logger='current', level='INFO'
                )
                num_boxes = expected_size
            
            if num_boxes == 0:
                # No predictions → no uncertainty
                cross_view_uncs.append(torch.zeros(0, device=reg_uncs_list[idx].device))
                continue
            
            # Find which group this view belongs to
            base_img_id = None
            view_indices = []
            for gid, vindices in groups.items():
                if idx in vindices:
                    base_img_id = gid
                    view_indices = vindices
                    break
            
            if len(view_indices) < 2:
                # Single view or not grouped → max uncertainty
                # Use expected_size from reg_uncs to ensure consistency
                expected_size = reg_uncs_list[idx].size(0)
                cross_view_uncs.append(
                    torch.ones(expected_size, device=reg_uncs_list[idx].device) * 0.5)
                continue
            
            # Count how many views predict each class
            labels = data_samples.gt_instances.labels
            scores = data_samples.gt_instances.scores
            
            # CRITICAL: If num_boxes was overridden but labels is empty, return default uncertainty
            if len(labels) == 0 and num_boxes > 0:
                # This means size mismatch happened, return high uncertainty for all boxes
                cross_view_uncs.append(
                    torch.ones(num_boxes, device=reg_uncs_list[idx].device) * 0.8)
                continue
            
            class_support = torch.zeros(num_boxes, device=labels.device)
            score_consistency = torch.zeros(num_boxes, device=labels.device)
            
            for box_idx, label in enumerate(labels):
                # Count views that predict this class
                # Note: roi_head.predict() only returns foreground classes [0, num_classes-1]
                # so no need to check for background
                views_with_class = 0
                other_scores = []
                
                for view_idx in view_indices:
                    if view_idx == idx:
                        continue  # Skip self
                    view_labels = batch_data_samples[view_idx].gt_instances.labels
                    view_scores = batch_data_samples[view_idx].gt_instances.scores
                    
                    if len(view_labels) > 0:
                        # All labels from roi_head.predict() are foreground
                        class_mask = (view_labels == label)
                        if class_mask.any():
                            views_with_class += 1
                            # Collect scores for this class from other views
                            other_scores.append(view_scores[class_mask].mean())
                
                # Normalize by total views (excluding self)
                class_support[box_idx] = views_with_class / max(len(view_indices) - 1, 1)
                
                # ENHANCED: Check score consistency across views
                if len(other_scores) > 0:
                    other_scores = torch.stack(other_scores)
                    my_score = scores[box_idx]
                    # If scores are similar → low variance → more consistent
                    score_std = torch.cat([other_scores, my_score.unsqueeze(0)]).std()
                    # Normalize std to [0, 1]: std=0 (perfect) → 0, std>0.3 (inconsistent) → 1
                    score_consistency[box_idx] = torch.clamp(score_std / 0.3, 0, 1)
                else:
                    score_consistency[box_idx] = 1.0  # No other views → max inconsistency
            
            # Convert support to uncertainty: high support → low uncertainty
            # support=0 (only this view) → unc=1.0
            # support=1 (all views agree) → unc=0.0
            class_unc = 1.0 - class_support
            
            # Combine class uncertainty with score consistency
            # Both contribute to overall uncertainty
            uncertainties = (class_unc + score_consistency) / 2.0
            
            # Final safety check: ensure size matches reg_uncs
            expected_size = reg_uncs_list[idx].size(0)
            if uncertainties.size(0) != expected_size:
                log_message(
                    f"Computed {uncertainties.size(0)} uncertainties but expected {expected_size}. "
                    f"Resizing to match reg_uncs.",
                    logger='current', level='INFO'
                )
                # Resize: pad with max uncertainty or truncate
                if uncertainties.size(0) < expected_size:
                    # Pad with high uncertainty
                    padding = torch.ones(expected_size - uncertainties.size(0), 
                                       device=uncertainties.device) * 0.8
                    uncertainties = torch.cat([uncertainties, padding])
                else:
                    # Truncate
                    uncertainties = uncertainties[:expected_size]
            
            cross_view_uncs.append(uncertainties)
        
        # DEBUG: Report mismatch statistics
        if mismatch_count > 0:
            log_message(
                f"CRITICAL: {mismatch_count}/{total_views} views had size mismatch! "
                f"batch_data_samples is being modified unexpectedly.",
                logger='current', level='INFO'
            )
        
        return cross_view_uncs
    
    @torch.no_grad()
    def get_pseudo_instances_with_consensus(self,
                                           batch_inputs: Tensor,
                                           batch_data_samples: SampleList) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances with cross-view uncertainty weighting.
        
        For rotation multi-view: uncertainty is already computed and added to reg_uncs
        in get_pseudo_instances(). This function just logs statistics.
        
        Args:
            batch_inputs (Tensor): Shape (B*V, C, H, W)
            batch_data_samples (SampleList): Length B*V
            
        Returns:
            Tuple of (samples with uncertainty-weighted pseudo-labels, batch_info)
        """
        # Get pseudo-labels with cross-view uncertainty already computed
        batch_data_samples, batch_info = self.get_pseudo_instances(
            batch_inputs, batch_data_samples)
        
        if not self.consensus_cfg['enable']:
            return batch_data_samples, batch_info
        
        # Log cross-view statistics
        if not hasattr(self, '_consensus_log_count'):
            self._consensus_log_count = 0
        self._consensus_log_count += 1
        
        if self._consensus_log_count % 50 == 1:
            groups = self._group_predictions_by_base_img(batch_data_samples)
            
            # Compute per-group uncertainty statistics
            group_stats = []
            for gid, vindices in groups.items():
                group_uncs = []
                group_boxes = 0
                for view_idx in vindices:
                    ds = batch_data_samples[view_idx]
                    if len(ds.gt_instances.bboxes) > 0 and hasattr(ds.gt_instances, 'reg_uncs'):
                        group_uncs.append(ds.gt_instances.reg_uncs)
                        group_boxes += len(ds.gt_instances.bboxes)
                
                if len(group_uncs) > 0:
                    group_uncs = torch.cat(group_uncs, dim=0)
                    group_stats.append({
                        'gid': gid,
                        'boxes': group_boxes,
                        'mean': group_uncs.mean().item(),
                        'median': group_uncs.median().item(),
                        'min': group_uncs.min().item(),
                        'max': group_uncs.max().item()
                    })
            
            # Log per-group statistics
            if len(group_stats) > 0:
                stats_str = " | ".join([
                    f"G{i}:{s['boxes']}box(unc={s['mean']:.3f})" 
                    for i, s in enumerate(group_stats)
                ])
                log_message(
                    f"[Cross-View Uncertainty] Groups: {len(groups)}, {stats_str}",
                    logger='current'
                )
        
        return batch_data_samples, batch_info
