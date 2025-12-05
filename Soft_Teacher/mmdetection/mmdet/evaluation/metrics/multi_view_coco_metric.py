# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-View COCO Metric for evaluating multi-view object detection.

This metric extends CocoMetric to handle multi-view predictions by:
1. Grouping predictions from multiple views of the same base image
2. Aggregating predictions within each group (NMS, voting, etc.)
3. Computing mAP on aggregated results
"""

import os.path as osp
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class MultiViewCocoMetric(CocoMetric):
    """Multi-View COCO detection metric.
    
    This metric groups predictions from multiple views of the same base image,
    aggregates them, and computes mAP on the aggregated results.
    
    Args:
        ann_file (str): Path to COCO annotation file
        metric (str): Metric to compute. Options are 'bbox'. Default: 'bbox'.
        views_per_sample (int): Number of views per sample. Default: 8.
        aggregation (str): How to aggregate multi-view predictions.
            Options: 'nms', 'soft_nms', 'wbf', 'voting'. Default: 'nms'.
        nms_iou_thr (float): IoU threshold for NMS aggregation. Default: 0.5.
        extract_base_name (bool): Whether to extract base image name from filename.
            If True, expects filenames like "base_crop_X.jpg". Default: True.
        **kwargs: Additional arguments for CocoMetric.
    
    Example:
        >>> # In config file
        >>> val_evaluator = dict(
        >>>     type='MultiViewCocoMetric',
        >>>     ann_file='data/anno_valid/_annotations.coco.json',
        >>>     metric='bbox',
        >>>     views_per_sample=8,
        >>>     aggregation='nms',
        >>>     nms_iou_thr=0.5,
        >>>     extract_base_name=True
        >>> )
    """
    
    def __init__(self,
                 ann_file: str,
                 metric: str = 'bbox',
                 views_per_sample: int = 8,
                 aggregation: str = 'nms',
                 nms_iou_thr: float = 0.5,
                 extract_base_name: bool = True,
                 enable_aggregation: bool = True,  # New: allow disabling aggregation
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            **kwargs
        )
        
        self.views_per_sample = views_per_sample
        self.aggregation = aggregation
        self.nms_iou_thr = nms_iou_thr
        self.extract_base_name = extract_base_name
        self.enable_aggregation = enable_aggregation
        
        assert aggregation in ['nms', 'soft_nms', 'wbf', 'voting', 'none'], \
            f"aggregation must be one of ['nms', 'soft_nms', 'wbf', 'voting', 'none'], got {aggregation}"
        
        # If aggregation is 'none' or enable_aggregation is False, skip aggregation
        if aggregation == 'none' or not enable_aggregation:
            self.enable_aggregation = False
            self.aggregation = 'none'
        
        self.logger = MMLogger.get_current_instance()
        if self.enable_aggregation:
            self.logger.info(
                f"MultiViewCocoMetric initialized: views_per_sample={views_per_sample}, "
                f"aggregation={aggregation}, nms_iou_thr={nms_iou_thr}"
            )
        else:
            self.logger.info(
                f"MultiViewCocoMetric initialized: PER-CROP evaluation mode "
                f"(no aggregation, {views_per_sample} crops evaluated independently)"
            )
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        The predictions are grouped by base image name before aggregation.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # DEBUG: Count process() calls
        if not hasattr(self, '_process_call_count'):
            self._process_call_count = 0
            self._total_samples = 0
        self._process_call_count += 1
        self._total_samples += len(data_samples)
        
        # if self._process_call_count <= 5 or self._process_call_count % 20 == 0:
        #     from mmengine.logging import print_log
        #     print_log(f"[DEBUG process] Call #{self._process_call_count}: {len(data_samples)} samples (total so far: {self._total_samples})",
        #              logger='current')
        #     if self._process_call_count <= 2:
        #         for i, ds in enumerate(data_samples[:3]):
        #             print_log(f"  Sample {i}: base_img_id={ds.get('base_img_id', 'N/A')}, img_id={ds.get('img_id', 'N/A')}",
        #                      logger='current')
        
        # First, collect all predictions (per-view) - same format as parent
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            
            # Store metadata for grouping AND coordinate transformation
            if 'img_path' in data_sample:
                result['img_path'] = data_sample['img_path']
            elif 'ori_filename' in data_sample:
                result['img_path'] = data_sample['ori_filename']
            else:
                result['img_path'] = f"img_{data_sample['img_id']}"
            
            # Store base_img_id for grouping (if available from dataset)
            if 'base_img_id' in data_sample:
                result['base_img_id'] = data_sample['base_img_id']
            
            # CRITICAL: Store homography matrix for projecting boxes to original space
            if 'homography_matrix' in data_sample:
                result['homography_matrix'] = data_sample['homography_matrix']
            else:
                # Identity matrix if not available (no transformation needed)
                result['homography_matrix'] = np.eye(3)
            
            # Parse gt (same as parent class)
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            
            # Store as tuple (gt, result) for compatibility with parent class
            self.results.append((gt, result))
    
    def evaluate(self, size: int) -> dict:
        """Override evaluate to handle multi-view flattening.
        
        The dataset reports size as number of groups, but we have
        views_per_sample views per group. Need to pass correct size
        to collect_results to avoid truncation.
        
        Args:
            size (int): Number of groups in dataset
            
        Returns:
            dict: Evaluation metrics
        """
        # Correct size: number of groups × views per group
        # This prevents collect_results from truncating our flattened views
        actual_size = size * self.views_per_sample
        
        from mmengine.logging import print_log
        print_log(f"[DEBUG evaluate] Dataset size={size} groups, " + 
                 f"actual_size={actual_size} views, " +
                 f"self.results has {len(self.results)} items",
                 logger='current')
        
        # Call parent with corrected size
        return super().evaluate(actual_size)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Multi-view evaluation with two modes:
        1. Aggregated: Group 8 views → 1 base image, compute mAP on 80 images
        2. Per-crop: Evaluate each crop independently, compute mAP on 640 crops
        
        Args:
            results (list): The processed results of each batch.
                Each element is a tuple (gt, result).
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        # DEBUG: Check results size
        from mmengine.logging import print_log
        print_log(f"[DEBUG compute_metrics] Received {len(results)} results", logger='current')
        print_log(f"[DEBUG compute_metrics] self.results has {len(self.results)} items", logger='current')
        
        # Split gt and prediction list
        gts, preds = zip(*results)
        
        # Mode 1: Per-crop evaluation (no aggregation)
        if not self.enable_aggregation:
            logger.info('\n' + '='*80)
            logger.info('Multi-View COCO Evaluation (Per-Crop Mode)')
            logger.info('='*80)
            logger.info(f"Evaluating {len(preds)} crops independently (no aggregation)")
            logger.info(f"MVViT cross-view attention still active during training!")
            logger.info('='*80 + '\n')
            
            # Call parent class directly on all crops
            metrics = super().compute_metrics(results)
            
            # Add multi-view info
            metrics['mv_evaluation_mode'] = 'per_crop'
            metrics['mv_num_crops_evaluated'] = len(preds)
            metrics['mv_aggregation_method'] = 'none'
            
            logger.info('\n' + '='*80)
            logger.info('Per-Crop Evaluation Summary')
            logger.info('='*80)
            logger.info(f"Total crops evaluated: {len(preds)}")
            logger.info(f"Note: This is per-crop mAP, not base-image mAP")
            logger.info('='*80 + '\n')
            
            return metrics
        
        # Mode 2: Aggregated evaluation (original behavior)
        logger.info('\n' + '='*80)
        logger.info('Multi-View COCO Evaluation (Aggregated Mode)')
        logger.info('='*80)
        
        # Split gt and prediction list
        gts, preds = zip(*results)
        
        logger.info(f"\n[Step 1] Grouping {len(preds)} view predictions by base image...")
        
        # Group predictions by base_img_id
        grouped_predictions = self._group_predictions(list(preds))
        logger.info(f"  - Number of base image groups: {len(grouped_predictions)}")
        logger.info(f"  - Views per group: {self.views_per_sample}")
        logger.info(f"  - Total views: {len(preds)}")
        
        # Aggregate predictions within each group (8 views → 1 base image)
        logger.info(f"\n[Step 2] Aggregating predictions using '{self.aggregation}'...")
        aggregated_preds = self._aggregate_predictions(grouped_predictions)
        logger.info(f"  - Aggregated to {len(aggregated_preds)} base images")
        
        # Get ground truth for base images
        logger.info(f"\n[Step 3] Preparing ground truth for base images...")
        base_gts = self._get_base_image_gts(grouped_predictions, gts)
        logger.info(f"  - Prepared GT for {len(base_gts)} base images")
        
        # Create aggregated results for parent class
        aggregated_results = list(zip(base_gts, aggregated_preds))
        
        # Compute metrics on aggregated base images (80 images, not 640 views)
        logger.info(f"\n[Step 4] Computing COCO metrics on {len(aggregated_results)} base images...")
        metrics = super().compute_metrics(aggregated_results)
        
        # Add multi-view specific info to metrics
        logger.info('\n' + '='*80)
        logger.info('Multi-View Summary')
        logger.info('='*80)
        logger.info(f"Evaluation mode: Aggregated (8 views → 1 base image)")
        logger.info(f"Aggregation method: {self.aggregation}")
        logger.info(f"Views per sample: {self.views_per_sample}")
        logger.info(f"Number of groups: {len(grouped_predictions)}")
        logger.info(f"Total views collected: {len(preds)}")
        logger.info(f"Base images evaluated: {len(aggregated_results)}")
        logger.info('='*80 + '\n')
        
        # Store multi-view info in metrics
        metrics['mv_num_views'] = self.views_per_sample
        metrics['mv_num_groups'] = len(grouped_predictions)
        metrics['mv_total_views_evaluated'] = len(preds)
        metrics['mv_base_images_evaluated'] = len(aggregated_results)
        metrics['mv_evaluation_mode'] = 'aggregated'
        metrics['mv_aggregation_method'] = self.aggregation
        
        return metrics
    
    def _extract_base_name(self, img_path: str) -> str:
        """Extract base image name from crop filename.
        
        Expects format like: "S140_Image__2025-09-24__10-40-29_dark_1_crop_7.jpg"
        Returns: "S140_Image__2025-09-24__10-40-29_dark_1"
        
        Args:
            img_path (str): Image path or filename
            
        Returns:
            str: Base image name
        """
        # Get filename without extension
        filename = osp.basename(img_path)
        name_without_ext = osp.splitext(filename)[0]
        
        # Extract base name (everything before _crop_X)
        if '_crop_' in name_without_ext:
            # Pattern: base_crop_X or base_crop_X_suffix
            match = re.match(r'(.+?)_crop_\d+', name_without_ext)
            if match:
                return match.group(1)
        
        # Fallback: use full name
        return name_without_ext
    
    def _group_predictions(self, results: list) -> Dict[str, List[dict]]:
        """Group predictions by base image name.
        
        Args:
            results (list): List of per-view predictions
            
        Returns:
            Dict[str, List[dict]]: Grouped predictions {base_name: [view_preds]}
        """
        grouped = defaultdict(list)
        
        # DEBUG: Log first 10 results to see what base_img_id values we're getting
        from mmengine.logging import print_log
        if not hasattr(self, '_grouping_debug_done'):
            self._grouping_debug_done = True
            print_log(f"[DEBUG _group_predictions] Total results: {len(results)}", logger='current')
            for i, result in enumerate(results[:10]):
                print_log(f"  Result {i}: base_img_id={result.get('base_img_id', 'N/A')}, " + 
                         f"img_id={result.get('img_id', 'N/A')}, img_path={result.get('img_path', 'N/A')[:50]}...",
                         logger='current')
        
        for result in results:
            # Priority 1: Use base_img_id if available (most reliable)
            if 'base_img_id' in result:
                base_name = str(result['base_img_id'])
            # Priority 2: Extract from filename pattern
            elif self.extract_base_name:
                base_name = self._extract_base_name(result['img_path'])
            # Priority 3: Fallback to img_id grouping
            else:
                # Use img_id grouping: assume consecutive views
                # img_id 0-7 -> group 0, 8-15 -> group 1, etc.
                group_idx = result['img_id'] // self.views_per_sample
                base_name = f"group_{group_idx}"
            
            grouped[base_name].append(result)
        
        return grouped
    
    def _get_base_image_gts(self, grouped_predictions: Dict[str, List[dict]], 
                            all_gts: tuple) -> list:
        """Aggregate ground truth from all views to base image.
        
        CRITICAL: Keep ALL GT annotations from all views (UNION, no deduplication).
        GT overlap between views is intentional - each view's GT is valid ground truth.
        
        We:
        1. Transform GT boxes from crop coords → base image coords (homography)
        2. Collect GT from all 8 views into one union (NO deduplication)
        3. Allow multiple overlapping GT boxes (legitimate multi-view annotations)
        
        Args:
            grouped_predictions: Grouped predictions by base_img_id
            all_gts: All GT dicts from results (640 GTs for crops)
            
        Returns:
            list: Union GT dicts for base images (80 GTs with all view annotations)
        """
        base_gts = []
        
        # Create a mapping from img_id to GT
        gt_dict = {gt['img_id']: gt for gt in all_gts}
        
        for base_name, view_preds in grouped_predictions.items():
            if len(view_preds) == 0:
                continue
            
            # Collect all GT annotations from all views (UNION)
            all_view_anns = []
            base_width = 2560  # Default base image size
            base_height = 1080
            base_img_id = view_preds[0]['img_id']  # Use first view's id
            
            for view_pred in view_preds:
                view_img_id = view_pred['img_id']
                
                if view_img_id not in gt_dict:
                    continue
                
                view_gt = gt_dict[view_img_id]
                homography = view_pred.get('homography_matrix')
                
                # Transform each GT annotation to base image coordinates
                for ann in view_gt.get('anns', []):
                    bbox = ann['bbox']  # [x, y, w, h] in crop coords
                    
                    if homography is not None:
                        # Transform bbox to base image coords
                        bbox_xyxy = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])
                        transformed = self._project_boxes_to_original_space(bbox_xyxy, homography)
                        
                        # Convert back to [x, y, w, h]
                        x1, y1, x2, y2 = transformed[0]
                        transformed_bbox = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        # No homography: keep original (shouldn't happen)
                        transformed_bbox = bbox
                    
                    # Add to union of all GT annotations (NO deduplication)
                    all_view_anns.append({
                        'bbox': transformed_bbox,
                        'category_id': ann['category_id'],
                        'area': transformed_bbox[2] * transformed_bbox[3],
                        'iscrowd': ann.get('iscrowd', 0),
                        'ignore': ann.get('ignore', 0)
                    })
                
                # Get base image size from first view with valid size
                if 'width' in view_gt and view_gt['width'] > base_width:
                    base_width = view_gt['width']
                if 'height' in view_gt and view_gt['height'] > base_height:
                    base_height = view_gt['height']
            
            # NO deduplication - keep all GT boxes from all views
            # This allows predictions to match any GT from any view
            
            # Create union GT for base image
            base_gts.append({
                'img_id': base_img_id,
                'width': int(base_width),
                'height': int(base_height),
                'anns': all_view_anns  # All annotations, no filtering
            })
        
        return base_gts
    
    # Deduplication removed - we keep ALL GT boxes from all views
    # This is intentional: GT overlap between views is legitimate ground truth
    
    def _project_boxes_to_original_space(self, bboxes: np.ndarray, 
                                         homography: np.ndarray) -> np.ndarray:
        """Project bounding boxes from crop space to original image space.
        
        CRITICAL: Without this projection, boxes from different crops have different
        coordinate systems, causing IoU=0 and breaking NMS/WBF/Voting aggregation.
        
        The homography matrix H transforms points from crop to original:
            [x_orig]   [H11 H12 H13]   [x_crop]
            [y_orig] = [H21 H22 H23] @ [y_crop]
            [   1  ]   [H31 H32 H33]   [  1   ]
        
        For bounding boxes in [x1, y1, x2, y2] format, we project all 4 corners
        and compute new axis-aligned bounding box.
        
        Args:
            bboxes (np.ndarray): (N, 4) boxes in crop space [x1, y1, x2, y2]
            homography (np.ndarray): (3, 3) homography matrix
            
        Returns:
            np.ndarray: (N, 4) boxes in original image space [x1, y1, x2, y2]
        """
        if len(bboxes) == 0:
            return bboxes
        
        # Check if identity matrix (no transformation needed)
        if np.allclose(homography, np.eye(3)):
            return bboxes
        
        N = len(bboxes)
        projected_bboxes = np.zeros_like(bboxes)
        
        for i in range(N):
            x1, y1, x2, y2 = bboxes[i]
            
            # Get 4 corners of bbox in crop space
            corners = np.array([
                [x1, y1, 1],  # top-left
                [x2, y1, 1],  # top-right
                [x2, y2, 1],  # bottom-right
                [x1, y2, 1],  # bottom-left
            ]).T  # (3, 4)
            
            # Project corners to original space: H @ corners
            projected_corners = homography @ corners  # (3, 4)
            
            # Convert from homogeneous to Cartesian coordinates
            projected_corners = projected_corners[:2] / projected_corners[2:]  # (2, 4)
            
            # Compute axis-aligned bbox from projected corners
            x_min = projected_corners[0].min()
            y_min = projected_corners[1].min()
            x_max = projected_corners[0].max()
            y_max = projected_corners[1].max()
            
            projected_bboxes[i] = [x_min, y_min, x_max, y_max]
        
        return projected_bboxes
    
    def _aggregate_predictions(self, grouped_predictions: Dict[str, List[dict]]) -> list:
        """Aggregate predictions within each group.
        
        CRITICAL: Projects boxes from crop coordinate space to original image space
        using homography_matrix BEFORE aggregation. Without this, IoU between boxes
        from different crops is 0 (different coordinate systems).
        
        Args:
            grouped_predictions (Dict[str, List[dict]]): Grouped predictions
            
        Returns:
            list: Aggregated results in same format as input
        """
        aggregated = []
        logger: MMLogger = MMLogger.get_current_instance()
        
        # Statistics for aggregation
        groups_with_detections = 0
        groups_without_detections = 0
        boxes_projected = 0
        
        # DEBUG: Track prediction statistics
        total_views_with_preds = 0
        total_views_without_preds = 0
        total_boxes_before_projection = 0
        
        for base_name, view_preds in grouped_predictions.items():
            if len(view_preds) == 0:
                continue
            
            # DEBUG: Log prediction stats for first few groups
            if groups_with_detections + groups_without_detections < 3:
                logger.info(f"[DEBUG Group {base_name}] {len(view_preds)} views")
                for i, pred in enumerate(view_preds):
                    has_boxes = len(pred.get('bboxes', [])) > 0
                    logger.info(f"  View {i}: {len(pred.get('bboxes', []))} boxes, scores shape: {pred.get('scores', []).shape}")
                    if has_boxes:
                        total_views_with_preds += 1
                    else:
                        total_views_without_preds += 1
                    total_boxes_before_projection += len(pred.get('bboxes', []))
            if len(view_preds) == 0:
                continue
            
            # Use first view's img_id for aggregated result
            # (or could create new synthetic img_id)
            base_img_id = view_preds[0]['img_id']
            
            # Collect all bboxes, scores, labels from all views
            # CRITICAL: Project boxes to original space BEFORE collecting
            all_bboxes = []
            all_scores = []
            all_labels = []
            
            for pred in view_preds:
                if len(pred['bboxes']) > 0:
                    # DEBUG: Check homography matrix
                    homo = pred.get('homography_matrix', np.eye(3))
                    has_real_homography = not np.allclose(homo, np.eye(3))
                    
                    if groups_with_detections + groups_without_detections < 3:
                        logger.info(f"[DEBUG Group {base_name}] View has homography: {has_real_homography}")
                        if has_real_homography:
                            logger.info(f"  Homography matrix:\n{homo}")
                        else:
                            logger.warning(f"[WARNING] No homography matrix for group {base_name} - using identity (no projection)")
                    
                    # PROJECT boxes from crop space to original image space
                    # TEMPORARY: Skip projection if no homography (use identity)
                    if has_real_homography:
                        projected_bboxes = self._project_boxes_to_original_space(
                            pred['bboxes'], 
                            homo
                        )
                    else:
                        # No projection - keep boxes as-is (WRONG but allows debugging)
                        projected_bboxes = pred['bboxes']
                        if groups_with_detections + groups_without_detections < 3:
                            logger.warning(f"[TEMPORARY FIX] Skipping projection for group {base_name} - boxes may be in wrong coordinate system")
                    boxes_projected += len(projected_bboxes)
                    
                    all_bboxes.append(projected_bboxes)
                    all_scores.append(pred['scores'])
                    all_labels.append(pred['labels'])
            
            if len(all_bboxes) == 0:
                # No detections in any view
                groups_without_detections += 1
                aggregated.append({
                    'img_id': base_img_id,
                    'bboxes': np.zeros((0, 4)),
                    'scores': np.zeros(0),
                    'labels': np.zeros(0, dtype=np.int64),
                    'img_path': base_name
                })
                continue
            
            groups_with_detections += 1
            
            # Concatenate all predictions (NOW in same coordinate space!)
            all_bboxes = np.vstack(all_bboxes)
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            
            # Apply aggregation (NMS/WBF/Voting will work correctly now)
            if self.aggregation == 'nms':
                agg_bboxes, agg_scores, agg_labels = self._nms_aggregation(
                    all_bboxes, all_scores, all_labels
                )
            elif self.aggregation == 'soft_nms':
                agg_bboxes, agg_scores, agg_labels = self._soft_nms_aggregation(
                    all_bboxes, all_scores, all_labels
                )
            elif self.aggregation == 'wbf':
                agg_bboxes, agg_scores, agg_labels = self._wbf_aggregation(
                    all_bboxes, all_scores, all_labels
                )
            elif self.aggregation == 'voting':
                agg_bboxes, agg_scores, agg_labels = self._voting_aggregation(
                    all_bboxes, all_scores, all_labels
                )
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
            
            # DEBUG: Log aggregation result
            if groups_with_detections < 3:
                logger.info(f"[DEBUG Group {base_name}] Aggregated: {len(agg_bboxes)} boxes")
            
            aggregated.append({
                'img_id': base_img_id,
                'bboxes': agg_bboxes,
                'scores': agg_scores,
                'labels': agg_labels,
                'img_path': base_name
            })
        
        # DEBUG: Final statistics
        logger.info(f"[DEBUG Aggregation Stats] Groups with detections: {groups_with_detections}")
        logger.info(f"[DEBUG Aggregation Stats] Groups without detections: {groups_without_detections}")
        logger.info(f"[DEBUG Aggregation Stats] Views with predictions: {total_views_with_preds}")
        logger.info(f"[DEBUG Aggregation Stats] Views without predictions: {total_views_without_preds}")
        logger.info(f"[DEBUG Aggregation Stats] Total boxes before projection: {total_boxes_before_projection}")
        logger.info(f"[DEBUG Aggregation Stats] Boxes projected: {boxes_projected}")
        
        return aggregated
        logger.info(f"  - Groups with detections: {groups_with_detections}")
        logger.info(f"  - Groups without detections: {groups_without_detections}")
        logger.info(f"  - Total boxes projected to original space: {boxes_projected}")
        
        return aggregated
    
    def _nms_aggregation(self, bboxes: np.ndarray, scores: np.ndarray, 
                        labels: np.ndarray) -> tuple:
        """Apply NMS to aggregate predictions.
        
        Args:
            bboxes (np.ndarray): (N, 4) bboxes in [x1, y1, x2, y2]
            scores (np.ndarray): (N,) confidence scores
            labels (np.ndarray): (N,) class labels
            
        Returns:
            tuple: (filtered_bboxes, filtered_scores, filtered_labels)
        """
        from mmdet.structures.bbox import bbox_overlaps
        
        # Apply NMS per class
        keep_all = []
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_bboxes) == 0:
                continue
            
            # Convert to torch for bbox_overlaps
            cls_bboxes_t = torch.from_numpy(cls_bboxes).float()
            cls_scores_t = torch.from_numpy(cls_scores).float()
            
            # Compute IoU matrix
            ious = bbox_overlaps(cls_bboxes_t, cls_bboxes_t)
            
            # Standard NMS
            keep = []
            order = cls_scores_t.argsort(descending=True)
            
            while order.numel() > 0:
                if order.numel() == 1:
                    i = order.item()
                    keep.append(i)
                    break
                
                i = order[0].item()
                keep.append(i)
                
                # Find boxes with IoU <= threshold (keep them)
                ovr = ious[order[0], order[1:]]
                inds = (ovr <= self.nms_iou_thr).nonzero(as_tuple=False).squeeze(1)
                
                if inds.numel() == 0:
                    break
                    
                order = order[inds + 1]
            
            # Convert local indices back to global
            keep_all.extend(cls_indices[keep].tolist())
        
        if len(keep_all) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        
        return bboxes[keep_all], scores[keep_all], labels[keep_all]
    
    def _soft_nms_aggregation(self, bboxes: np.ndarray, scores: np.ndarray,
                             labels: np.ndarray) -> tuple:
        """Apply Soft-NMS with Gaussian decay.
        
        Instead of removing boxes, Soft-NMS decays their scores based on IoU overlap.
        Better for overlapping objects in multi-view scenarios.
        
        Args:
            bboxes (np.ndarray): (N, 4) boxes in [x1, y1, x2, y2]
            scores (np.ndarray): (N,) confidence scores
            labels (np.ndarray): (N,) class labels
            
        Returns:
            tuple: (filtered_bboxes, filtered_scores, filtered_labels)
        """
        from mmdet.structures.bbox import bbox_overlaps
        
        keep_all = []
        keep_scores_all = []
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask].copy()  # Copy for modification
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_bboxes) == 0:
                continue
            
            # Convert to torch for bbox_overlaps
            cls_bboxes_t = torch.from_numpy(cls_bboxes).float()
            
            # Compute IoU matrix
            ious = bbox_overlaps(cls_bboxes_t, cls_bboxes_t).numpy()
            
            # Soft-NMS with Gaussian decay
            kept_indices = []
            kept_scores = []
            sigma = 0.5  # Gaussian width parameter
            
            while len(cls_scores) > 0:
                # Find box with highest score
                max_idx = cls_scores.argmax()
                kept_indices.append(max_idx)
                kept_scores.append(cls_scores[max_idx])
                
                # Decay scores of overlapping boxes
                ovr = ious[max_idx]
                weight = np.exp(-(ovr * ovr) / sigma)
                cls_scores = cls_scores * weight
                cls_scores[max_idx] = 0  # Remove selected box
            
            # Convert local indices back to global
            keep_all.extend(cls_indices[kept_indices].tolist())
            keep_scores_all.extend(kept_scores)
        
        if len(keep_all) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        
        return bboxes[keep_all], np.array(keep_scores_all), labels[keep_all]
    
    def _wbf_aggregation(self, bboxes: np.ndarray, scores: np.ndarray,
                        labels: np.ndarray) -> tuple:
        """Apply Weighted Box Fusion to aggregate multi-view predictions.
        
        WBF averages overlapping boxes weighted by their confidence scores.
        Best for multi-view as it utilizes ALL views instead of just picking one.
        
        Algorithm:
        1. Find clusters of overlapping boxes (IoU > threshold)
        2. For each cluster, compute weighted average of coordinates and scores
        3. Final bbox = weighted avg, final score = avg of cluster scores
        
        Args:
            bboxes (np.ndarray): (N, 4) boxes in [x1, y1, x2, y2]
            scores (np.ndarray): (N,) confidence scores
            labels (np.ndarray): (N,) class labels
            
        Returns:
            tuple: (fused_bboxes, fused_scores, fused_labels)
        """
        from mmdet.structures.bbox import bbox_overlaps
        
        fused_bboxes_all = []
        fused_scores_all = []
        fused_labels_all = []
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            if len(cls_bboxes) == 0:
                continue
            
            # Convert to torch for bbox_overlaps
            cls_bboxes_t = torch.from_numpy(cls_bboxes).float()
            cls_scores_t = torch.from_numpy(cls_scores).float()
            
            # Compute IoU matrix
            ious = bbox_overlaps(cls_bboxes_t, cls_bboxes_t).numpy()
            
            # Find clusters of overlapping boxes
            used = np.zeros(len(cls_bboxes), dtype=bool)
            
            for i in range(len(cls_bboxes)):
                if used[i]:
                    continue
                
                # Find all boxes overlapping with box i
                cluster_mask = ious[i] > self.nms_iou_thr
                cluster_indices = np.where(cluster_mask)[0]
                
                # Mark as used
                used[cluster_indices] = True
                
                # Get cluster boxes and scores
                cluster_boxes = cls_bboxes[cluster_indices]
                cluster_scores = cls_scores[cluster_indices]
                
                # Weighted Box Fusion
                # Weights are normalized scores
                weights = cluster_scores / cluster_scores.sum()
                
                # Weighted average of coordinates
                fused_box = np.sum(cluster_boxes * weights[:, None], axis=0)
                
                # Average score (could also use max or weighted avg)
                fused_score = cluster_scores.mean()
                
                fused_bboxes_all.append(fused_box)
                fused_scores_all.append(fused_score)
                fused_labels_all.append(cls)
        
        if len(fused_bboxes_all) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        
        return (np.array(fused_bboxes_all), 
                np.array(fused_scores_all), 
                np.array(fused_labels_all, dtype=np.int64))
    
    def _voting_aggregation(self, bboxes: np.ndarray, scores: np.ndarray,
                           labels: np.ndarray) -> tuple:
        """Apply box voting to aggregate predictions.
        
        Voting finds consensus boxes by clustering overlapping detections.
        Only keeps boxes that have votes from multiple views (robust to outliers).
        
        Args:
            bboxes (np.ndarray): (N, 4) boxes in [x1, y1, x2, y2]
            scores (np.ndarray): (N,) confidence scores
            labels (np.ndarray): (N,) class labels
            
        Returns:
            tuple: (voted_bboxes, voted_scores, voted_labels)
        """
        from mmdet.structures.bbox import bbox_overlaps
        
        voted_bboxes_all = []
        voted_scores_all = []
        voted_labels_all = []
        
        min_votes = 2  # Minimum number of views that must agree
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            if len(cls_bboxes) == 0:
                continue
            
            # Convert to torch for bbox_overlaps
            cls_bboxes_t = torch.from_numpy(cls_bboxes).float()
            
            # Compute IoU matrix
            ious = bbox_overlaps(cls_bboxes_t, cls_bboxes_t).numpy()
            
            # Find clusters and count votes
            used = np.zeros(len(cls_bboxes), dtype=bool)
            
            for i in range(len(cls_bboxes)):
                if used[i]:
                    continue
                
                # Find all boxes overlapping with box i
                cluster_mask = ious[i] > self.nms_iou_thr
                cluster_indices = np.where(cluster_mask)[0]
                num_votes = len(cluster_indices)
                
                # Only keep if enough votes (consensus from multiple views)
                if num_votes < min_votes:
                    used[cluster_indices] = True
                    continue
                
                # Mark as used
                used[cluster_indices] = True
                
                # Get cluster boxes and scores
                cluster_boxes = cls_bboxes[cluster_indices]
                cluster_scores = cls_scores[cluster_indices]
                
                # Vote: median of coordinates (robust to outliers)
                voted_box = np.median(cluster_boxes, axis=0)
                
                # Score: max score from cluster
                voted_score = cluster_scores.max()
                
                voted_bboxes_all.append(voted_box)
                voted_scores_all.append(voted_score)
                voted_labels_all.append(cls)
        
        if len(voted_bboxes_all) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        
        return (np.array(voted_bboxes_all), 
                np.array(voted_scores_all), 
                np.array(voted_labels_all, dtype=np.int64))
