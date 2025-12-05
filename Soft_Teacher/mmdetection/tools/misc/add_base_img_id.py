#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Add base_img_id field to COCO annotation files for multi-view datasets.

This script adds base_img_id and crop_num fields to each image entry in a COCO JSON file.
The base_img_id is used to group multiple views (crops) of the same sample together.

Multi-view images are identified by their filename pattern:
    S123_Image__2025-09-23__15-38-40_dark_2_crop_3_jpg.rf.xxx.jpg
    ↓
    base_img_id = "S123_dark_2" (sample_id + lighting + camera)
    crop_num = 3

Usage:
    # Single file
    python tools/misc/add_base_img_id.py \\
        --input data_drill/semi_anno_multiview/_annotations.coco.labeled.grouped@40.json \\
        --output data_drill/semi_anno_multiview/_annotations.coco.labeled.grouped@40.json
    
    # Multiple files (in-place update)
    python tools/misc/add_base_img_id.py \\
        --input data_drill/semi_anno_multiview/_annotations.coco.*.json \\
        --in-place
"""

import argparse
import glob
import json
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add base_img_id field to COCO annotation files')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input COCO JSON file(s). Supports glob patterns like "*.json"')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path. If not specified, will append ".updated" to input filename')
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Update files in-place (overwrite input files)')
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .bak backup before in-place update')
    args = parser.parse_args()
    return args


def extract_base_img_id(filename):
    """Extract base_img_id and crop_num from multi-view filename.
    
    Args:
        filename (str): Image filename like 'S123_Image__..._dark_2_crop_3_jpg.rf.xxx.jpg'
        
    Returns:
        tuple: (base_img_id, crop_num) or (None, None) if pattern doesn't match
        
    Examples:
        >>> extract_base_img_id('S225_Image__2025-07-01__14-30-51_bright_2_crop_7_jpg.rf.xxx.jpg')
        ('S225_bright_2', 7)
        >>> extract_base_img_id('S136_Image__2025-09-23__16-42-55_dark_4_crop_3_jpg.rf.xxx.jpg')
        ('S136_dark_4', 3)
    """
    # Pattern: S{num}_*_{lighting}_{camera}_crop_{crop_num}
    # Groups by: S{num}_{lighting}_{camera}
    pattern = r'^(S\d+)_.*_(bright|dark)_(\d+)_crop_(\d+)'
    match = re.match(pattern, filename)
    
    if match:
        sample_id = match.group(1)   # S225
        lighting = match.group(2)     # bright or dark
        camera = match.group(3)       # 2, 3, 4, 5, etc.
        crop_num = match.group(4)     # 1, 2, 3, ..., 8
        
        base_img_id = f"{sample_id}_{lighting}_{camera}"
        return base_img_id, int(crop_num)
    else:
        # Pattern doesn't match - might be different dataset format
        return None, None


def add_base_img_id_to_coco(input_path, output_path, backup=False):
    """Add base_img_id and crop_num fields to COCO annotation file.
    
    Args:
        input_path (str): Path to input COCO JSON file
        output_path (str): Path to output COCO JSON file
        backup (bool): Create backup before overwriting (if input == output)
        
    Returns:
        dict: Statistics about the update
    """
    print(f'\nProcessing: {input_path}')
    
    # Load COCO JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    if not images:
        print(f'  Warning: No images found in {input_path}')
        return {'total': 0, 'updated': 0, 'skipped': 0}
    
    # Add base_img_id and crop_num to each image
    updated_count = 0
    skipped_count = 0
    
    for img in images:
        filename = img['file_name']
        base_img_id, crop_num = extract_base_img_id(filename)
        
        if base_img_id is not None:
            img['base_img_id'] = base_img_id
            img['crop_num'] = crop_num
            updated_count += 1
        else:
            # Pattern didn't match - skip this image
            skipped_count += 1
            if skipped_count <= 3:  # Show first 3 warnings
                print(f'  Warning: Could not extract base_img_id from: {filename}')
    
    # Create backup if requested
    if backup and input_path == output_path:
        backup_path = input_path + '.bak'
        os.rename(input_path, backup_path)
        print(f'  Created backup: {backup_path}')
    
    # Save updated JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print statistics
    print(f'  Total images: {len(images)}')
    print(f'  Updated: {updated_count}')
    print(f'  Skipped: {skipped_count}')
    
    # Verify grouping
    from collections import defaultdict, Counter
    groups = defaultdict(list)
    for img in images:
        base_id = img.get('base_img_id')
        if base_id:
            groups[base_id].append(img.get('crop_num'))
    
    if groups:
        print(f'  Unique groups: {len(groups)}')
        crops_per_group = Counter(len(crops) for crops in groups.values())
        print(f'  Crops per group distribution: {dict(sorted(crops_per_group.items()))}')
        
        # Show first 2 groups as examples
        print(f'  Example groups:')
        for i, (base_id, crops) in enumerate(list(groups.items())[:2]):
            sorted_crops = sorted(crops)
            print(f'    {base_id}: {len(crops)} crops (nums: {sorted_crops})')
    
    print(f'  Saved to: {output_path}')
    
    return {
        'total': len(images),
        'updated': updated_count,
        'skipped': skipped_count,
        'groups': len(groups)
    }


def main():
    args = parse_args()
    
    # Expand glob patterns
    input_files = glob.glob(args.input)
    if not input_files:
        print(f'Error: No files found matching pattern: {args.input}')
        return
    
    print('=' * 80)
    print('Add base_img_id to COCO Multi-View Annotations')
    print('=' * 80)
    print(f'Input pattern: {args.input}')
    print(f'Found {len(input_files)} file(s)')
    print('=' * 80)
    
    total_stats = {
        'files': 0,
        'total_images': 0,
        'updated_images': 0,
        'skipped_images': 0,
        'total_groups': 0
    }
    
    for input_path in input_files:
        # Determine output path
        if args.in_place:
            output_path = input_path
        elif args.output:
            if len(input_files) > 1:
                # Multiple files but single output specified - not allowed
                print(f'Error: Cannot specify single --output with multiple input files')
                print(f'       Use --in-place instead, or process files one by one')
                return
            output_path = args.output
        else:
            # Append .updated before extension
            base, ext = os.path.splitext(input_path)
            output_path = f'{base}.updated{ext}'
        
        # Process file
        stats = add_base_img_id_to_coco(input_path, output_path, backup=args.backup)
        
        # Accumulate statistics
        total_stats['files'] += 1
        total_stats['total_images'] += stats['total']
        total_stats['updated_images'] += stats['updated']
        total_stats['skipped_images'] += stats['skipped']
        total_stats['total_groups'] += stats['groups']
    
    # Print summary
    print('\n' + '=' * 80)
    print('Summary')
    print('=' * 80)
    print(f"Files processed: {total_stats['files']}")
    print(f"Total images: {total_stats['total_images']}")
    print(f"Updated: {total_stats['updated_images']}")
    print(f"Skipped: {total_stats['skipped_images']}")
    print(f"Total groups: {total_stats['total_groups']}")
    print('=' * 80)
    print('✅ Done!')


if __name__ == '__main__':
    main()
