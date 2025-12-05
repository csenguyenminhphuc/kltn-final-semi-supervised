#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""K-Fold COCO split for Multi-View datasets with grouped samples.

This script splits multi-view COCO annotations while keeping all views of the same
sample together (either all in labeled or all in unlabeled set).

Multi-view images are identified by their base name pattern:
    - Original: S123_Image__2025-09-23__15-38-40_dark_2.jpg
    - Crops: S123_Image__2025-09-23__15-38-40_dark_2_crop_1.jpg
             S123_Image__2025-09-23__15-38-40_dark_2_crop_2.jpg
             ...
             S123_Image__2025-09-23__15-38-40_dark_2_crop_8.jpg

All these images belong to the same "group" and will be kept together.

Usage:
    python tools/misc/split_coco_multiview_grouped.py \\
        --data-root ./data_drill/ \\
        --ann-file anno_train/_annotations.coco.json \\
        --out-dir semi_anns/ \\
        --labeled-percent 20 40 \\
        --fold 5
"""

import argparse
import json
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np


prog_description = '''K-Fold multi-view coco split with grouped samples.

To split multi-view coco data for semi-supervised object detection:
    python tools/misc/split_coco_multiview_grouped.py \\
        --data-root ./data_drill/ \\
        --ann-file anno_train/_annotations.coco.json \\
        --labeled-percent 20 40
'''


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='./data_drill/')
    parser.add_argument(
        '--ann-file',
        type=str,
        help='The annotation file relative to data-root.',
        default='anno_train/_annotations.coco.json')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.',
        default='./data_drill/semi_anns/')
    parser.add_argument(
        '--labeled-percent',
        type=float,
        nargs='+',
        help='The percentage of labeled data in the training set.',
        default=[20, 40])
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=5)
    parser.add_argument(
        '--views-per-sample',
        type=int,
        help='Number of views (crops) per sample. Use -1 for auto-detection.',
        default=8)
    args = parser.parse_args()
    return args


def extract_base_name(filename):
    """Extract base name from multi-view filename.
    
    Groups images by sample ID + lighting + camera angle.
    All crops from the same (sample, lighting, camera) are one group.
    
    Examples:
        S123_Image__2025-09-23__15-38-40_dark_2_crop_3_jpg.rf.xxx 
            -> S123_dark_2
        S123_Image__2025-09-23__15-38-40_bright_4_crop_7_jpg.rf.xxx 
            -> S123_bright_4
    
    Pattern: S{num}_*_{lighting}_{camera}_crop_{X}_*
    Groups by: S{num}_{lighting}_{camera}
    """
    # Pattern to extract: sample_id, lighting (bright/dark), camera number
    # Example: S225_Image__2025-07-01__14-30-51_bright_2_crop_7_jpg.rf.xxx
    pattern = r'^(S\d+)_.*_(bright|dark)_(\d+)_crop_\d+'
    match = re.match(pattern, filename)
    
    if match:
        sample_id = match.group(1)   # S225
        lighting = match.group(2)     # bright or dark
        camera = match.group(3)       # 2, 3, 4, etc.
        return f"{sample_id}_{lighting}_{camera}"
    else:
        # Fallback: use old logic if pattern doesn't match
        base = osp.splitext(filename)[0]
        base = re.sub(r'\.rf\.[a-f0-9]+$', '', base)
        base = re.sub(r'_crop_\d+$', '', base)
        base = re.sub(r'_jpg$', '', base)
        return base


def group_images_by_sample(images):
    """Group images by their base sample name.
    
    Args:
        images (list): List of image dicts with 'id' and 'file_name'.
        
    Returns:
        dict: Mapping from base_name to list of image dicts.
    """
    groups = defaultdict(list)
    for img in images:
        base_name = extract_base_name(img['file_name'])
        groups[base_name].append(img)
    return groups


def split_coco_multiview(data_root, ann_file, out_dir, percent, fold, views_per_sample=-1):
    """Split COCO data for Semi-supervised object detection with multi-view grouping.

    Args:
        data_root (str): The data root of coco dataset.
        ann_file (str): The annotation file relative to data_root.
        out_dir (str): The output directory of coco semi-supervised annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
        views_per_sample (int): Number of views per sample. -1 for auto-detection.
    """

    def save_anns(name, images, annotations):
        sub_anns = dict()
        sub_anns['images'] = images
        sub_anns['annotations'] = annotations
        if 'licenses' in anns:
            sub_anns['licenses'] = anns['licenses']
        sub_anns['categories'] = anns['categories']
        if 'info' in anns:
            sub_anns['info'] = anns['info']

        os.makedirs(out_dir, exist_ok=True)
        output_path = osp.join(out_dir, f'{name}.json')
        with open(output_path, 'w') as f:
            json.dump(sub_anns, f, indent=2)
        print(f'Saved {name}: {len(images)} images, {len(annotations)} annotations')

    # Set random seed with the fold
    np.random.seed(fold)
    
    ann_path = osp.join(data_root, ann_file)
    print(f'Loading annotations from: {ann_path}')
    with open(ann_path, 'r') as f:
        anns = json.load(f)

    image_list = anns['images']
    print(f'Total images: {len(image_list)}')
    
    # Group images by base sample name
    image_groups = group_images_by_sample(image_list)
    group_names = list(image_groups.keys())
    print(f'Total unique samples (groups): {len(group_names)}')
    
    # Auto-detect views per sample if needed
    if views_per_sample == -1:
        view_counts = [len(imgs) for imgs in image_groups.values()]
        views_per_sample = int(np.median(view_counts))
        print(f'Auto-detected views_per_sample: {views_per_sample}')
    
    # Verify grouping
    view_counts = defaultdict(int)
    for imgs in image_groups.values():
        view_counts[len(imgs)] += 1
    print(f'View count distribution: {dict(view_counts)}')
    
    # Split groups (not individual images)
    labeled_total_groups = int(percent / 100. * len(group_names))
    print(f'Selecting {labeled_total_groups} groups ({percent}%) as labeled')
    
    labeled_group_inds = set(
        np.random.choice(range(len(group_names)), size=labeled_total_groups, replace=False))
    
    labeled_ids, labeled_images, unlabeled_images = set(), [], []
    
    for i, group_name in enumerate(group_names):
        group_images = image_groups[group_name]
        if i in labeled_group_inds:
            # All views of this sample go to labeled set
            labeled_images.extend(group_images)
            for img in group_images:
                labeled_ids.add(img['id'])
        else:
            # All views of this sample go to unlabeled set
            unlabeled_images.extend(group_images)
    
    print(f'Labeled images: {len(labeled_images)} ({len(labeled_images)/len(image_list)*100:.1f}%)')
    print(f'Unlabeled images: {len(unlabeled_images)} ({len(unlabeled_images)/len(image_list)*100:.1f}%)')
    
    # Get annotations for labeled and unlabeled images
    labeled_annotations, unlabeled_annotations = [], []
    
    for ann in anns['annotations']:
        if ann['image_id'] in labeled_ids:
            labeled_annotations.append(ann)
        else:
            unlabeled_annotations.append(ann)
    
    print(f'Labeled annotations: {len(labeled_annotations)}')
    print(f'Unlabeled annotations: {len(unlabeled_annotations)}')
    
    # Save labeled and unlabeled splits
    labeled_name = f'_annotations.coco.labeled.grouped@{int(percent)}'
    unlabeled_name = f'_annotations.coco.unlabeled.grouped@{int(percent)}'
    
    save_anns(labeled_name, labeled_images, labeled_annotations)
    save_anns(unlabeled_name, unlabeled_images, unlabeled_annotations)
    
    print(f'Split completed for fold {fold}, percent {percent}%')
    print('-' * 80)


def multi_wrapper(args):
    return split_coco_multiview(*args)


if __name__ == '__main__':
    args = parse_args()
    
    print('=' * 80)
    print('Multi-View COCO Dataset Split with Grouped Samples')
    print('=' * 80)
    print(f'Data root: {args.data_root}')
    print(f'Annotation file: {args.ann_file}')
    print(f'Output directory: {args.out_dir}')
    print(f'Labeled percentages: {args.labeled_percent}')
    print(f'Number of folds: {args.fold}')
    print(f'Views per sample: {args.views_per_sample}')
    print('=' * 80)
    
    # For multi-view grouped split, we typically use single fold
    # since we're grouping by samples, not doing k-fold CV on images
    for percent in args.labeled_percent:
        print(f'\nProcessing {percent}% labeled split...')
        split_coco_multiview(
            args.data_root, 
            args.ann_file, 
            args.out_dir, 
            percent, 
            fold=1,  # Use fold=1 as default for grouped split
            views_per_sample=args.views_per_sample
        )
    
    print('\n' + '=' * 80)
    print('All splits completed successfully!')
    print('=' * 80)
