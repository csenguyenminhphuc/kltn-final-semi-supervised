"""Count bounding boxes per group in multi-view COCO JSON."""

import json
from collections import defaultdict
import numpy as np

# Load labeled dataset
json_path = 'data_drill/semi_anno_multiview/_annotations.coco.unlabeled.grouped@60.bright.json'

print("="*80)
print(f"Analyzing: {json_path}")
print("="*80)

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"\nTotal images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")

# Count boxes per group
group_boxes = defaultdict(list)  # group_id -> [crop1_boxes, crop2_boxes, ...]
group_crop_boxes = defaultdict(lambda: defaultdict(int))  # group_id -> {crop_num: box_count}

# Build image lookup
img_lookup = {img['id']: img for img in data['images']}

for ann in data['annotations']:
    img_id = ann['image_id']
    
    # Find image info
    img_info = img_lookup.get(img_id)
    if not img_info:
        continue
    
    base_img_id = img_info.get('base_img_id', img_id // 8)  # Fallback if not present
    crop_num = img_info.get('crop_num', (img_id % 8) + 1)
    
    group_crop_boxes[base_img_id][crop_num] += 1
    group_boxes[base_img_id].append(crop_num)

# Calculate statistics
print("\n" + "="*80)
print("Group Statistics")
print("="*80)

total_groups = len(group_crop_boxes)
boxes_per_group = []
crops_per_group = []

for group_id in sorted(group_crop_boxes.keys()):
    crop_counts = group_crop_boxes[group_id]
    total_boxes = sum(crop_counts.values())
    num_crops = len(crop_counts)
    
    boxes_per_group.append(total_boxes)
    crops_per_group.append(num_crops)

print(f"\nTotal groups: {total_groups}")
print(f"Average boxes per group: {np.mean(boxes_per_group):.2f} ± {np.std(boxes_per_group):.2f}")
print(f"Min boxes per group: {np.min(boxes_per_group)}")
print(f"Max boxes per group: {np.max(boxes_per_group)}")
print(f"Median boxes per group: {np.median(boxes_per_group):.1f}")

print(f"\nAverage crops per group: {np.mean(crops_per_group):.2f}")
print(f"Groups with all 8 crops: {sum(1 for c in crops_per_group if c == 8)}")

# Distribution
print("\n" + "="*80)
print("Boxes Per Group Distribution")
print("="*80)

bins = [0, 5, 10, 15, 20, 30, 50, 100]
hist, _ = np.histogram(boxes_per_group, bins=bins + [np.inf])

for i, (low, high) in enumerate(zip(bins, bins[1:] + [np.inf])):
    if high == np.inf:
        print(f"{low}+ boxes: {hist[i]} groups ({hist[i]/total_groups*100:.1f}%)")
    else:
        print(f"{low}-{high-1} boxes: {hist[i]} groups ({hist[i]/total_groups*100:.1f}%)")

# Show groups with < 5 boxes
print("\n" + "="*80)
print("Groups with < 5 Total Boxes")
print("="*80)
print(f"{'Base ID':<30} {'Total Boxes':<15} {'# Crops':<10} Boxes per crop")
print("-"*80)

low_box_groups = []
for group_id in sorted(group_crop_boxes.keys()):
    crop_counts = group_crop_boxes[group_id]
    total_boxes = sum(crop_counts.values())
    
    if total_boxes < 5:
        num_crops = len(crop_counts)
        crop_str = " | ".join([f"C{crop}:{count}" for crop, count in sorted(crop_counts.items())])
        print(f"{group_id:<30} {total_boxes:<15} {num_crops:<10} {crop_str}")
        low_box_groups.append(group_id)

print(f"\nTotal groups with < 5 boxes: {len(low_box_groups)}")

# Show first 20 groups in detail
print("\n" + "="*80)
print("Groups by Base Image (First 20)")
print("="*80)
print(f"{'Base ID':<30} {'Total Boxes':<15} {'# Crops':<10} Boxes per crop")
print("-"*80)

for group_id in sorted(group_crop_boxes.keys())[:20]:
    crop_counts = group_crop_boxes[group_id]
    total_boxes = sum(crop_counts.values())
    num_crops = len(crop_counts)
    
    # Format boxes per crop
    crop_str = " | ".join([f"C{crop}:{count}" for crop, count in sorted(crop_counts.items())])
    
    print(f"{group_id:<30} {total_boxes:<15} {num_crops:<10} {crop_str}")

# Per-crop statistics
print("\n" + "="*80)
print("Per-Crop Statistics")
print("="*80)

crop_totals = defaultdict(int)
crop_groups = defaultdict(int)

for group_id, crop_counts in group_crop_boxes.items():
    for crop_num, count in crop_counts.items():
        crop_totals[crop_num] += count
        crop_groups[crop_num] += 1

print(f"\n{'Crop':<8} {'Groups':<10} {'Total Boxes':<15} {'Avg Boxes/Group':<20}")
print("-"*80)

for crop_num in sorted(crop_totals.keys()):
    total = crop_totals[crop_num]
    groups = crop_groups[crop_num]
    avg = total / groups if groups > 0 else 0
    print(f"{crop_num:<8} {groups:<10} {total:<15} {avg:.2f}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"✅ Total groups: {total_groups}")
print(f"✅ Total boxes: {sum(boxes_per_group)}")
print(f"✅ Average: {np.mean(boxes_per_group):.1f} boxes/group")
print(f"✅ Each group has {np.mean(crops_per_group):.1f} crops on average")
print("="*80)
