#!/usr/bin/env python3
"""Analyze bbox sizes in COCO dataset and recommend anchor scales/ratios.

This script analyzes bounding box dimensions to help configure:
1. RPN anchor scales and ratios
2. min_bbox_size threshold
3. FilterAnnotations parameters

Usage:
    python tools/misc/analyze_bbox_for_anchors.py \\
        --ann-file data_drill/anno_train/_annotations_filtered.coco.json \\
        --img-size 720 256 \\
        --output-file bbox_analysis.txt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze bbox sizes for anchor configuration')
    parser.add_argument(
        '--ann-file',
        type=str,
        required=True,
        help='Path to COCO annotation file')
    parser.add_argument(
        '--img-size',
        type=int,
        nargs=2,
        default=[720, 256],
        help='Image size (height, width) after resize. Default: 720 256')
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file to save analysis results')
    parser.add_argument(
        '--show-samples',
        type=int,
        default=10,
        help='Number of sample bboxes to display')
    args = parser.parse_args()
    return args


def analyze_bboxes(ann_file, img_size=(720, 256)):
    """Analyze bbox statistics from COCO annotation file.
    
    Args:
        ann_file: Path to COCO annotation JSON
        img_size: (height, width) tuple for target image size
        
    Returns:
        dict: Statistics about bbox sizes
    """
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    # Extract bbox dimensions
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for ann in coco['annotations']:
        bbox = ann['bbox']  # [x, y, w, h]
        w, h = bbox[2], bbox[3]
        
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        
        # Aspect ratio: width / height
        if h > 0:
            aspect_ratios.append(w / h)
    
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)
    
    # Compute statistics
    stats = {
        'total_annotations': len(widths),
        'width': {
            'min': widths.min(),
            'max': widths.max(),
            'mean': widths.mean(),
            'median': np.median(widths),
            'std': widths.std(),
            'percentiles': {
                1: np.percentile(widths, 1),
                5: np.percentile(widths, 5),
                10: np.percentile(widths, 10),
                25: np.percentile(widths, 25),
                50: np.percentile(widths, 50),
                75: np.percentile(widths, 75),
                90: np.percentile(widths, 90),
                95: np.percentile(widths, 95),
                99: np.percentile(widths, 99),
            }
        },
        'height': {
            'min': heights.min(),
            'max': heights.max(),
            'mean': heights.mean(),
            'median': np.median(heights),
            'std': heights.std(),
            'percentiles': {
                1: np.percentile(heights, 1),
                5: np.percentile(heights, 5),
                10: np.percentile(heights, 10),
                25: np.percentile(heights, 25),
                50: np.percentile(heights, 50),
                75: np.percentile(heights, 75),
                90: np.percentile(heights, 90),
                95: np.percentile(heights, 95),
                99: np.percentile(heights, 99),
            }
        },
        'area': {
            'min': areas.min(),
            'max': areas.max(),
            'mean': areas.mean(),
            'median': np.median(areas),
        },
        'aspect_ratio': {
            'min': aspect_ratios.min(),
            'max': aspect_ratios.max(),
            'mean': aspect_ratios.mean(),
            'median': np.median(aspect_ratios),
            'std': aspect_ratios.std(),
            'percentiles': {
                1: np.percentile(aspect_ratios, 1),
                5: np.percentile(aspect_ratios, 5),
                10: np.percentile(aspect_ratios, 10),
                25: np.percentile(aspect_ratios, 25),
                50: np.percentile(aspect_ratios, 50),
                75: np.percentile(aspect_ratios, 75),
                90: np.percentile(aspect_ratios, 90),
                95: np.percentile(aspect_ratios, 95),
                99: np.percentile(aspect_ratios, 99),
            }
        }
    }
    
    # Count small boxes
    small_counts = {
        'w<4 or h<4': np.sum((widths < 4) | (heights < 4)),
        'w<8 or h<8': np.sum((widths < 8) | (heights < 8)),
        'w<16 or h<16': np.sum((widths < 16) | (heights < 16)),
        'w<32 or h<32': np.sum((widths < 32) | (heights < 32)),
    }
    stats['small_boxes'] = small_counts
    
    return stats, widths, heights, aspect_ratios


def recommend_anchors(stats, img_size=(720, 256)):
    """Recommend anchor scales and ratios based on bbox statistics.
    
    Args:
        stats: Statistics dict from analyze_bboxes
        img_size: (height, width) tuple
        
    Returns:
        dict: Recommended anchor configurations
    """
    h_img, w_img = img_size
    
    # FPN strides
    strides = [4, 8, 16, 32, 64]
    
    # Recommend scales based on bbox size distribution
    # Use median width as reference
    median_w = stats['width']['median']
    median_h = stats['height']['median']
    
    # Typical FPN stride for median object: find stride where median_size/stride ≈ 8-16
    ref_size = (median_w + median_h) / 2
    ref_stride = ref_size / 8  # Target ~8 pixels per anchor
    
    # Recommend base scale
    base_scale = 8  # Standard anchor base size
    
    # Recommend ratios based on aspect ratio distribution
    ar_percentiles = stats['aspect_ratio']['percentiles']
    
    # Aspect ratio candidates: width/height
    # Add more fine-grained ratios for narrow objects
    ratios_candidates = [0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    # Select ratios that cover 5th to 95th percentile + margins
    ar_min = ar_percentiles[5]   # 5th percentile
    ar_max = ar_percentiles[95]  # 95th percentile
    ar_median = ar_percentiles[50]  # Median
    
    # Strategy: Select ratios that provide good coverage
    # 1. Always include ratios around median (most common shape)
    # 2. Include extremes to cover 5%-95% range
    # 3. Limit total number to avoid too many anchors
    
    recommended_ratios = []
    
    # Add ratios covering the data range with some margin
    for r in ratios_candidates:
        # Include if within data range (with 20% margin on both sides)
        if ar_min * 0.8 <= r <= ar_max * 1.2:
            recommended_ratios.append(r)
    
    # Always include 1.0 (square) for robustness
    if 1.0 not in recommended_ratios:
        recommended_ratios.append(1.0)
    
    # Sort and ensure we don't have too many ratios (max 7)
    recommended_ratios = sorted(set(recommended_ratios))
    
    if len(recommended_ratios) > 7:
        # Keep ratios that are most representative
        # Priority: median, extremes, and evenly spaced in between
        keep_indices = [0, len(recommended_ratios)//4, len(recommended_ratios)//2, 
                       3*len(recommended_ratios)//4, len(recommended_ratios)-1]
        # Add median if not already included
        median_idx = min(range(len(recommended_ratios)), 
                        key=lambda i: abs(recommended_ratios[i] - ar_median))
        keep_indices.append(median_idx)
        # Add 1.0 if exists
        if 1.0 in recommended_ratios:
            keep_indices.append(recommended_ratios.index(1.0))
        
        keep_indices = sorted(set(keep_indices))[:7]
        recommended_ratios = [recommended_ratios[i] for i in keep_indices]
    
    # Recommend min_bbox_size
    p1_w = stats['width']['percentiles'][1]
    p1_h = stats['height']['percentiles'][1]
    
    if p1_w < 4 or p1_h < 4:
        min_bbox_size = 0
    elif p1_w < 8 or p1_h < 8:
        min_bbox_size = 4
    elif p1_w < 16 or p1_h < 16:
        min_bbox_size = 8
    else:
        min_bbox_size = 16
    
    return {
        'scales': [base_scale],
        'ratios': recommended_ratios,
        'strides': strides,
        'min_bbox_size': min_bbox_size,
        'reference_size': ref_size,
        'ar_range': (ar_min, ar_median, ar_max),
    }


def print_report(stats, recommendations, output_file=None):
    """Print analysis report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("BBOX SIZE ANALYSIS REPORT")
    lines.append("=" * 80)
    
    lines.append(f"\nTotal annotations: {stats['total_annotations']}")
    
    # Width statistics
    lines.append("\n" + "=" * 80)
    lines.append("WIDTH STATISTICS (pixels)")
    lines.append("=" * 80)
    w = stats['width']
    lines.append(f"  Min:    {w['min']:.1f}")
    lines.append(f"  Max:    {w['max']:.1f}")
    lines.append(f"  Mean:   {w['mean']:.1f}")
    lines.append(f"  Median: {w['median']:.1f}")
    lines.append(f"  Std:    {w['std']:.1f}")
    lines.append("\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        lines.append(f"    {p:2d}%: {w['percentiles'][p]:6.1f}")
    
    # Height statistics
    lines.append("\n" + "=" * 80)
    lines.append("HEIGHT STATISTICS (pixels)")
    lines.append("=" * 80)
    h = stats['height']
    lines.append(f"  Min:    {h['min']:.1f}")
    lines.append(f"  Max:    {h['max']:.1f}")
    lines.append(f"  Mean:   {h['mean']:.1f}")
    lines.append(f"  Median: {h['median']:.1f}")
    lines.append(f"  Std:    {h['std']:.1f}")
    lines.append("\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        lines.append(f"    {p:2d}%: {h['percentiles'][p]:6.1f}")
    
    # Aspect ratio statistics
    lines.append("\n" + "=" * 80)
    lines.append("ASPECT RATIO STATISTICS (width / height)")
    lines.append("=" * 80)
    ar = stats['aspect_ratio']
    lines.append(f"  Min:    {ar['min']:.3f}")
    lines.append(f"  Max:    {ar['max']:.3f}")
    lines.append(f"  Mean:   {ar['mean']:.3f}")
    lines.append(f"  Median: {ar['median']:.3f}")
    lines.append(f"  Std:    {ar['std']:.3f}")
    lines.append("\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        lines.append(f"    {p:2d}%: {ar['percentiles'][p]:6.3f}")
    
    # Small boxes count
    lines.append("\n" + "=" * 80)
    lines.append("SMALL BBOX COUNTS")
    lines.append("=" * 80)
    for threshold, count in stats['small_boxes'].items():
        pct = count / stats['total_annotations'] * 100
        lines.append(f"  {threshold:15s}: {count:5d} boxes ({pct:5.2f}%)")
    
    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDED ANCHOR CONFIGURATION")
    lines.append("=" * 80)
    rec = recommendations
    
    lines.append("\nRPN Head Config:")
    lines.append("```python")
    lines.append("detector.rpn_head = dict(")
    lines.append("    type='RPNHead',")
    lines.append("    in_channels=256,")
    lines.append("    feat_channels=256,")
    lines.append("    anchor_generator=dict(")
    lines.append("        type='AnchorGenerator',")
    lines.append(f"        scales={rec['scales']},")
    lines.append(f"        ratios={rec['ratios']},")
    lines.append(f"        strides={rec['strides']}),")
    lines.append("    bbox_coder=dict(")
    lines.append("        type='DeltaXYWHBBoxCoder',")
    lines.append("        target_means=[.0, .0, .0, .0],")
    lines.append("        target_stds=[1.0, 1.0, 1.0, 1.0]),")
    lines.append("    loss_cls=dict(")
    lines.append("        type='CrossEntropyLoss',")
    lines.append("        use_sigmoid=True,")
    lines.append("        loss_weight=1.0),")
    lines.append("    loss_bbox=dict(type='L1Loss', loss_weight=1.0))")
    lines.append("```")
    
    lines.append(f"\nRecommended min_bbox_size: {rec['min_bbox_size']}")
    lines.append(f"Median object size: {rec['reference_size']:.1f} pixels")
    
    lines.append("\n" + "=" * 80)
    lines.append("EXPLANATION")
    lines.append("=" * 80)
    lines.append("\n1. **Scales**: [8] is the standard base anchor size")
    lines.append("   - Combined with 5 FPN strides [4,8,16,32,64], creates anchors:")
    lines.append("     * Stride 4:  32x32 base anchor (covers small objects)")
    lines.append("     * Stride 8:  64x64 base anchor")
    lines.append("     * Stride 16: 128x128 base anchor")
    lines.append("     * Stride 32: 256x256 base anchor")
    lines.append("     * Stride 64: 512x512 base anchor (covers large objects)")
    
    lines.append(f"\n2. **Ratios**: {rec['ratios']}")
    lines.append("   - Covers aspect ratio range of your dataset:")
    ar_min, ar_median, ar_max = rec['ar_range']
    lines.append(f"   - Dataset aspect ratios: {ar_min:.3f} (5%) to {ar_max:.3f} (95%), median={ar_median:.3f}")
    lines.append("   - Each ratio creates different anchor shapes:")
    for ratio in rec['ratios']:
        if ratio < 1:
            h_rel = 1.0 / ratio
            lines.append(f"     * {ratio:.2f}: Tall objects (h = {h_rel:.1f}w)")
        elif ratio == 1:
            lines.append(f"     * {ratio:.2f}: Square objects (h = w)")
        else:
            lines.append(f"     * {ratio:.2f}: Wide objects (w = {ratio:.1f}h)")
    
    # Add coverage analysis
    lines.append("\n   Coverage analysis:")
    ar = stats['aspect_ratio']
    all_ratios = np.array(rec['ratios'])
    percentiles_to_check = [5, 25, 50, 75, 95]
    
    for p in percentiles_to_check:
        ar_p = ar['percentiles'][p]
        # Find closest ratio
        closest_idx = np.argmin(np.abs(all_ratios - ar_p))
        closest_ratio = all_ratios[closest_idx]
        diff_pct = abs(closest_ratio - ar_p) / ar_p * 100
        lines.append(f"     * {p}% percentile ({ar_p:.3f}): closest ratio={closest_ratio:.2f} (diff={diff_pct:.1f}%)")
    
    lines.append(f"\n3. **min_bbox_size**: {rec['min_bbox_size']}")
    lines.append(f"   - Filters proposals smaller than {rec['min_bbox_size']}x{rec['min_bbox_size']} pixels")
    lines.append(f"   - Your smallest GT box: {w['percentiles'][1]:.1f}x{h['percentiles'][1]:.1f} (1st percentile)")
    lines.append("   - Safe threshold that removes false positives without losing GT")
    
    lines.append("\n" + "=" * 80)
    
    # Print to stdout
    report = "\n".join(lines)
    print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {output_file}")


def main():
    args = parse_args()
    
    # Analyze bboxes
    stats, widths, heights, aspect_ratios = analyze_bboxes(
        args.ann_file,
        img_size=tuple(args.img_size)
    )
    
    # Generate recommendations
    recommendations = recommend_anchors(stats, img_size=tuple(args.img_size))
    
    # Print report
    print_report(stats, recommendations, output_file=args.output_file)


if __name__ == '__main__':
    main()
