"""Filter multi-view COCO JSON to keep only bright images."""

import json
import sys

def filter_bright_images(input_file, output_file):
    """Filter to keep only images with 'bright' in filename."""
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Filter images: keep only 'bright' in filename
    bright_images = [img for img in data['images'] if 'bright' in img['file_name'].lower()]
    bright_img_ids = set(img['id'] for img in bright_images)
    
    print(f"Filtered: {len(bright_images)} bright images")
    
    # Filter annotations: keep only those belonging to bright images
    bright_annotations = [ann for ann in data['annotations'] if ann['image_id'] in bright_img_ids]
    
    print(f"Filtered: {len(bright_annotations)} annotations")
    
    # Create new dataset
    filtered_data = {
        'images': bright_images,
        'annotations': bright_annotations,
        'categories': data['categories'],
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }
    
    # Save
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✅ Done!")
    
    # Statistics
    base_images = set()
    for img in bright_images:
        base_name = img.get('base_img_id', img['file_name'].rsplit('_', 1)[0])
        base_images.add(base_name)
    
    print(f"\n📊 Statistics:")
    print(f"  - {len(base_images)} unique base images")
    print(f"  - {len(bright_images)} total crops")
    print(f"  - {len(bright_images) / len(base_images):.1f} crops per base image (avg)")
    print(f"  - {len(bright_annotations)} total annotations")
    print(f"  - {len(bright_annotations) / len(base_images):.1f} annotations per base image (avg)")

if __name__ == '__main__':
    # Filter labeled dataset
    print("="*80)
    print("Filtering LABELED dataset (60% split)")
    print("="*80)
    filter_bright_images(
        '/home/coder/data/trong/KLTN/Soft_Teacher/data_drill/anno_test/_annotations_filtered.coco.json',
        '/home/coder/data/trong/KLTN/Soft_Teacher/data_drill/anno_test/_annotations_filtered.bright.coco.json'
    )
    
    print("\n" + "="*80)
    # print("Filtering UNLABELED dataset (40% split)")
    # print("="*80)
    # filter_bright_images(
    #     'data_drill/semi_anno_multiview/_annotations.coco.unlabeled.grouped@60.json',
    #     'data_drill/semi_anno_multiview/_annotations.coco.unlabeled.grouped@60.bright.json'
    # )
    
    print("\n" + "="*80)
    print("✅ ALL DONE! Created 1 new files:")
    # print("  1. _annotations.coco.labeled.grouped@60.bright.json")
    # print("  2. _annotations.coco.unlabeled.grouped@60.bright.json")
    print("="*80)
