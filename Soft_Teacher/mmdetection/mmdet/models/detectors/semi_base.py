# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class SemiBaseDetector(BaseDetector):
    """Base class for semi-supervised detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

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
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(detector)
        self.teacher = MODELS.build(detector)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        
        # Handle supervised loss if labeled data exists in batch
        if 'sup' in multi_batch_inputs and 'sup' in multi_batch_data_samples:
            losses.update(**self.loss_by_gt_instances(
                multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        # Handle unsupervised loss if unlabeled data exists
        if 'unsup_teacher' in multi_batch_inputs and 'unsup_student' in multi_batch_inputs:
            # Use consensus-based pseudo-label generation if available
            if hasattr(self, 'get_pseudo_instances_with_consensus'):
                origin_pseudo_data_samples, batch_info = self.get_pseudo_instances_with_consensus(
                    multi_batch_inputs['unsup_teacher'],
                    multi_batch_data_samples['unsup_teacher'])
            else:
                origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
                    multi_batch_inputs['unsup_teacher'],
                    multi_batch_data_samples['unsup_teacher'])
            multi_batch_data_samples[
                'unsup_student'] = self.project_pseudo_instances(
                    origin_pseudo_data_samples,
                    multi_batch_data_samples['unsup_student'])
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'], batch_info))
        
        return losses

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        
        # Debug: Log GT labels in compact format
        from mmengine.logging import print_log
        if hasattr(self, '_debug_step_counter'):
            self._debug_step_counter += 1
        else:
            self._debug_step_counter = 0
        
        # Log first 10 iterations, then every 50 iterations
        should_log = self._debug_step_counter < 10 or self._debug_step_counter % 50 == 0
        
        if should_log:
            # Count GT labels across all samples
            classes = ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear')
            num_samples = len(batch_data_samples)
            
            # Determine views_per_sample (for multi-view)
            views_per_sample = getattr(self, 'views_per_sample', 8)
            num_groups = num_samples // views_per_sample
            
            # VERIFY: Check if samples have base_img_id to confirm grouping
            has_base_img_id = any(hasattr(ds, 'base_img_id') for ds in batch_data_samples[:min(3, num_samples)])
            
            if num_groups > 0:
                # Multi-view: log per-group statistics
                group_stats = []
                for g in range(num_groups):
                    start_idx = g * views_per_sample
                    end_idx = start_idx + views_per_sample
                    group_samples = batch_data_samples[start_idx:end_idx]
                    
                    # Verify same base_img_id within group
                    base_ids = [getattr(ds, 'base_img_id', None) for ds in group_samples]
                    unique_bases = set(b for b in base_ids if b is not None)
                    
                    group_counts = {cls: 0 for cls in classes}
                    group_boxes = 0
                    
                    for ds in group_samples:
                        gt_insts = ds.gt_instances
                        labels = gt_insts.labels if hasattr(gt_insts, 'labels') else []
                        if len(labels) > 0:
                            group_boxes += len(labels)
                            for label in labels.cpu().numpy():
                                label_int = int(label)
                                if label_int < len(classes):
                                    group_counts[classes[label_int]] += 1
                    
                    # Show warning if group has multiple base_img_ids (grouping broken)
                    if len(unique_bases) > 1:
                        group_stats.append(f"G{g}:MIXED!")
                    else:
                        group_stats.append(f"G{g}:{group_boxes}box")
                
                # One-line summary
                grouping_status = "✓" if has_base_img_id else "?"
                print_log(f"[Supervised Step {self._debug_step_counter}] Groups: {num_groups} {grouping_status}, {' | '.join(group_stats)}", logger='current')
            else:
                # Single-view or unknown: log total
                total_gt_counts = {cls: 0 for cls in classes}
                total_boxes = 0
                
                for ds in batch_data_samples:
                    gt_insts = ds.gt_instances
                    labels = gt_insts.labels if hasattr(gt_insts, 'labels') else []
                    if len(labels) > 0:
                        total_boxes += len(labels)
                        for label in labels.cpu().numpy():
                            label_int = int(label)
                            if label_int < len(classes):
                                total_gt_counts[classes[label_int]] += 1
                
                gt_str = ', '.join([f"{cls}:{cnt}" for cls, cnt in total_gt_counts.items() if cnt > 0])
                if total_boxes > 0:
                    print_log(f"[Supervised Step {self._debug_step_counter}] Samples: {num_samples}, GT: {total_boxes} boxes ({gt_str})", logger='current')
                else:
                    print_log(f"[Supervised Step {self._debug_step_counter}] Samples: {num_samples}, GT: 0 boxes", logger='current')

        losses = self.student.loss(batch_inputs, batch_data_samples)
        
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        
        # Debug: Log pseudo labels before filtering
        from mmengine.logging import print_log
        should_log = self._debug_step_counter < 10 or self._debug_step_counter % 50 == 0
        
        if should_log:
            classes = ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear')
            total_pseudo_counts = {cls: 0 for cls in classes}
            total_pseudo_before = 0
            
            for ds in batch_data_samples:
                gt_insts = ds.gt_instances
                labels = gt_insts.labels if hasattr(gt_insts, 'labels') else []
                scores = gt_insts.scores if hasattr(gt_insts, 'scores') else []
                if len(labels) > 0:
                    total_pseudo_before += len(labels)
                    for label in labels.cpu().numpy():
                        label_int = int(label)
                        if label_int < len(classes):
                            total_pseudo_counts[classes[label_int]] += 1
            
            pseudo_str = ', '.join([f"{cls}:{cnt}" for cls, cnt in total_pseudo_counts.items() if cnt > 0]) if total_pseudo_before > 0 else "None"
            print_log(f"[Unsupervised Step {self._debug_step_counter}] Pseudo (before cls filter, thr={self.semi_train_cfg.cls_pseudo_thr}): {total_pseudo_before} boxes ({pseudo_str})", logger='current')
        
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        
        # Debug: Log pseudo labels after filtering
        if should_log:
            classes = ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear')
            total_pseudo_counts_after = {cls: 0 for cls in classes}
            total_pseudo_after = 0
            
            for ds in batch_data_samples:
                gt_insts = ds.gt_instances
                labels = gt_insts.labels if hasattr(gt_insts, 'labels') else []
                if len(labels) > 0:
                    total_pseudo_after += len(labels)
                    for label in labels.cpu().numpy():
                        label_int = int(label)
                        if label_int < len(classes):
                            total_pseudo_counts_after[classes[label_int]] += 1
            
            kept_ratio = (total_pseudo_after / max(total_pseudo_before, 1)) * 100
            pseudo_str_after = ', '.join([f"{cls}:{cnt}" for cls, cnt in total_pseudo_counts_after.items() if cnt > 0]) if total_pseudo_after > 0 else "None"
            print_log(f"[Unsupervised Step {self._debug_step_counter}] Pseudo (after cls filter): {total_pseudo_after}/{total_pseudo_before} ({kept_ratio:.1f}%) boxes ({pseudo_str_after})", logger='current')
        
        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
        results_list = self.teacher.predict(
            batch_inputs, batch_data_samples, rescale=False)
        
        # Debug: Log teacher raw predictions
        from mmengine.logging import print_log
        if not hasattr(self, '_debug_teacher_pred_counter'):
            self._debug_teacher_pred_counter = 0
        self._debug_teacher_pred_counter += 1
        
        if self._debug_teacher_pred_counter <= 3 or self._debug_teacher_pred_counter % 50 == 0:
            total_preds = sum([len(r.pred_instances.bboxes) for r in results_list])
            classes = ('Broken','Chipped','Scratched','Severe_Rust','Tip_Wear')
            pred_counts = {cls: 0 for cls in classes}
            max_scores = []
            
            for r in results_list:
                if len(r.pred_instances.bboxes) > 0:
                    max_scores.extend(r.pred_instances.scores.cpu().numpy().tolist())
                    for label in r.pred_instances.labels.cpu().numpy():
                        if int(label) < len(classes):
                            pred_counts[classes[int(label)]] += 1
            
            pred_str = ', '.join([f"{cls}:{cnt}" for cls, cnt in pred_counts.items() if cnt > 0])
            max_score = max(max_scores) if max_scores else 0.0
            print_log(f"[Teacher Pred #{self._debug_teacher_pred_counter}] Raw predictions: {total_preds} boxes ({pred_str}), max_score={max_score:.3f}", logger='current')
        
        batch_info = {}
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results.pred_instances
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            results = self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            results = self.student(
                batch_inputs, batch_data_samples, mode='predict')
        
        # Debug: print predictions per batch (first validation batch only)
        if not hasattr(self, '_debug_pred_printed'):
            self._debug_pred_printed = True
            print(f"\n[DEBUG PREDICT] Batch size: {len(results)}")
            for i, r in enumerate(results[:3]):  # First 3 samples
                img_id = r.get('img_id', 'N/A')
                n_pred = len(r.pred_instances.bboxes) if hasattr(r, 'pred_instances') else 0
                labels = r.pred_instances.labels.tolist() if n_pred > 0 else []
                from collections import Counter
                label_counts = Counter(labels)
                print(f"  Sample {i}: img_id={img_id}, predictions={n_pred}, classes={dict(label_counts)}")
        
        return results

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        else:
            return self.student.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
