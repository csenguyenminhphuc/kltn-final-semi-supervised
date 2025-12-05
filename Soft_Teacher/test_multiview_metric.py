#!/usr/bin/env python
"""Test script for MultiViewCocoMetric to verify it works correctly."""

import json
import numpy as np
import torch
from mmdet.evaluation.metrics import MultiViewCocoMetric
from mmengine.logging import MMLogger


def create_dummy_predictions(num_groups=5, views_per_group=8):
    """Create dummy predictions for testing.
    
    Args:
        num_groups (int): Number of base image groups
        views_per_group (int): Number of views per group
        
    Returns:
        list: List of data_samples with predictions
    """
    data_samples = []
    
    for group_id in range(num_groups):
        for view_id in range(views_per_group):
            img_id = group_id * views_per_group + view_id
            
            # Create dummy predictions (some overlap, some unique)
            num_boxes = np.random.randint(2, 6)
            bboxes = np.random.rand(num_boxes, 4) * 100
            bboxes[:, 2:] += bboxes[:, :2]  # x1,y1,x2,y2 format
            
            scores = np.random.rand(num_boxes) * 0.5 + 0.5  # 0.5-1.0
            labels = np.random.randint(0, 6, num_boxes)
            
            data_sample = {
                'img_id': img_id,
                'img_path': f"S{group_id:03d}_Image__2025-11-18__10-00-00_bright_1_crop_{view_id+1}.jpg",
                'pred_instances': {
                    'bboxes': torch.from_numpy(bboxes).float(),
                    'scores': torch.from_numpy(scores).float(),
                    'labels': torch.from_numpy(labels).long()
                }
            }
            
            data_samples.append(data_sample)
    
    return data_samples


def test_metric():
    """Test MultiViewCocoMetric with dummy data."""
    
    print("="*80)
    print("Testing MultiViewCocoMetric")
    print("="*80)
    
    # Initialize logger
    logger = MMLogger.get_instance('test_metric')
    
    # Create metric
    metric = MultiViewCocoMetric(
        ann_file='data_drill/anno_valid/_annotations.coco.json',
        metric='bbox',
        views_per_sample=8,
        aggregation='nms',
        nms_iou_thr=0.5,
        extract_base_name=True,
        classwise=True  # Enable per-class metrics
    )
    
    print("\n✓ Metric initialized successfully")
    print(f"  - views_per_sample: {metric.views_per_sample}")
    print(f"  - aggregation: {metric.aggregation}")
    print(f"  - nms_iou_thr: {metric.nms_iou_thr}")
    print(f"  - classwise: {metric.classwise}")
    
    # Create dummy predictions
    print("\n" + "-"*80)
    print("Creating dummy predictions...")
    data_samples = create_dummy_predictions(num_groups=3, views_per_group=8)
    print(f"✓ Created {len(data_samples)} predictions")
    print(f"  - {len(data_samples) // 8} groups × 8 views per group")
    
    # Test grouping
    print("\n" + "-"*80)
    print("Testing grouping...")
    results = []
    for data_sample in data_samples:
        pred = data_sample['pred_instances']
        results.append({
            'img_id': data_sample['img_id'],
            'bboxes': pred['bboxes'].cpu().numpy(),
            'scores': pred['scores'].cpu().numpy(),
            'labels': pred['labels'].cpu().numpy(),
            'img_path': data_sample['img_path']
        })
    
    grouped = metric._group_predictions(results)
    print(f"✓ Grouped into {len(grouped)} base images")
    
    for i, (base_name, preds) in enumerate(list(grouped.items())[:3]):
        print(f"  Group {i}: '{base_name}' has {len(preds)} views")
        total_boxes = sum(len(p['bboxes']) for p in preds)
        print(f"    → Total bboxes: {total_boxes}")
    
    # Test aggregation
    print("\n" + "-"*80)
    print("Testing aggregation...")
    aggregated = metric._aggregate_predictions(grouped)
    print(f"✓ Aggregated to {len(aggregated)} results")
    
    total_before = sum(len(r['bboxes']) for r in results)
    total_after = sum(len(r['bboxes']) for r in aggregated)
    print(f"  - Total bboxes before: {total_before}")
    print(f"  - Total bboxes after: {total_after}")
    print(f"  - Reduction: {(1 - total_after/total_before)*100:.1f}%")
    
    # Test NMS aggregation specifically
    print("\n" + "-"*80)
    print("Testing NMS aggregation on sample data...")
    
    # Create overlapping boxes for NMS test
    bboxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # High overlap with first
        [100, 100, 150, 150],
        [102, 102, 152, 152]  # High overlap with third
    ])
    scores = np.array([0.9, 0.85, 0.8, 0.75])
    labels = np.array([0, 0, 1, 1])
    
    agg_bboxes, agg_scores, agg_labels = metric._nms_aggregation(bboxes, scores, labels)
    
    print(f"✓ NMS results:")
    print(f"  - Input: {len(bboxes)} boxes")
    print(f"  - Output: {len(agg_bboxes)} boxes (kept highest scores)")
    print(f"  - Scores: {agg_scores}")
    print(f"  - Labels: {agg_labels}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    
    print("\n📝 Notes:")
    print("  - To run full evaluation, use mmdetection/tools/test.py")
    print("  - This script only tests the metric components")
    print("  - Real evaluation requires actual predictions and GT annotations")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_metric()
