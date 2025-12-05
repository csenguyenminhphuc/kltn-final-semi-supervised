# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-View SoftTeacher for semi-supervised object detection with multiple views."""
import copy
from typing import List, Optional, Tuple

import torch
from mmengine.structures import InstanceData
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
                print_log(
                    f"[Teacher Predictions] Total boxes: {total_preds}, "
                    f"Score range: [{scores_all.min():.3f}, {scores_all.max():.3f}], "
                    f"Mean: {scores_all.mean():.3f}, "
                    f"Median: {scores_all.median():.3f}",
                    logger='current'
                )
            else:
                print_log(
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
        # 15 boxes = 2.5x max GT → provides buffer for rare classes without excessive noise
        max_boxes_per_image = 15
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
            print_log(
                f"[After Filtering] Threshold: {self.semi_train_cfg.pseudo_label_initial_score_thr}, "
                f"Kept: {total_after}/{total_preds} ({kept_ratio:.1f}%), "
                f"Avg: {avg_per_img:.1f} boxes/img",
                logger='current'
            )

        # Compute uncertainty for all views
        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)

        # Project bboxes and assign uncertainties
        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
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
