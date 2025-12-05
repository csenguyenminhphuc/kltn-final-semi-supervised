import json
import re

# Đọc file annotation gốc
input_file = '/home/coder/data/trong/KLTN/Soft_Teacher/data_drill/anno_test/_annotations.coco.json'
output_file = '/home/coder/data/trong/KLTN/Soft_Teacher/data_drill/anno_test/_annotations_filtered.coco.json'

print("Đang đọc file annotation...")
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"Số ảnh ban đầu: {len(data['images'])}")
print(f"Số annotations ban đầu: {len(data['annotations'])}")

# Tìm các ảnh cần loại bỏ (có index 1: bright_1, dark_1 hoặc crop_9)
pattern = re.compile(r'(bright_1|dark_1|crop_9)')
images_to_remove = set()

for img in data['images']:
    filename = img['file_name']
    # Kiểm tra xem filename có chứa bright_1, dark_1 hoặc crop_9
    if pattern.search(filename):
        images_to_remove.add(img['id'])
        print(f"Loại bỏ: {filename}")

print(f"\nSố ảnh cần loại bỏ: {len(images_to_remove)}")

# Lọc ảnh
filtered_images = [img for img in data['images'] if img['id'] not in images_to_remove]

# Lọc annotations tương ứng
filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] not in images_to_remove]

print(f"Số ảnh sau khi lọc: {len(filtered_images)}")
print(f"Số annotations sau khi lọc: {len(filtered_annotations)}")

# Tạo dữ liệu mới
filtered_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'categories': data['categories'],
    'images': filtered_images,
    'annotations': filtered_annotations
}

# Ghi file mới
print(f"\nĐang ghi file: {output_file}")
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print("Hoàn thành!")
