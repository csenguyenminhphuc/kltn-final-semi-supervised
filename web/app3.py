"""
Flask Web Application for MixPL Object Detection
Author: Khanh Phuc
Description: Web interface for running MixPL semi-supervised object detection model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import uuid
from datetime import datetime

# Thêm đường dẫn để import mmdet
sys.path.insert(0, '/home/coder/data/trong/KLTN/DETR_Mixup/MixPL')

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS

# Register all modules
register_all_modules()

# Flask app
app = Flask(__name__)

# Cấu hình đường dẫn
config_file = '/home/coder/data/trong/KLTN/DETR_Mixup/MixPL/projects/MixPL/configs/mixpl_coco_r50_custom_40.py'
checkpoint_file = '/home/coder/data/trong/KLTN/DETR_Mixup/MixPL/projects/MixPL/work_dirs/mixpl_coco_r50_custom_40/iter_60000.pth'

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Class names cho dataset drill bit defect
CLASS_NAMES = ['Broken', 'Chipped', 'Scratched', 'Severe_Rust', 'Tip_Wear']

# Class colors (RGB)
CLASS_COLORS = [
    (134, 34, 255),    # Broken - Tím
    (0, 255, 206),     # Chipped - Cyan
    (199, 252, 0),     # Scratched - Vàng xanh
    (254, 0, 86),      # Severe_Rust - Đỏ hồng
    (255, 128, 0),     # Tip_Wear - Cam
]

print(f"Config: {config_file}")
print(f"Checkpoint: {checkpoint_file}")
print(f"Device: {device}")

# Global model variable
model = None


def init_model():
    """Khởi tạo MixPL model"""
    global model
    try:
        print("Loading MixPL model...")
        
        # Load config
        cfg = Config.fromfile(config_file)
        
        # Build model
        model = MODELS.build(cfg.model)
        
        # Load checkpoint
        checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)
        
        # Thêm cfg vào model để inference_detector hoạt động
        model.cfg = cfg
        
        # Set model to eval mode
        model.eval()
        model.to(device)
        
        print("MixPL Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model on device: {next(model.parameters()).device}")
        print(f"Class names: {CLASS_NAMES}")
        
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def inference_mixpl(img_path):
    """
    Inference wrapper cho MixPL model
    
    Args:
        img_path: Đường dẫn đến ảnh cần inference
        
    Returns:
        Detection result
    """
    global model
    
    # MixPL sử dụng teacher model để inference
    # Lấy detector từ model (có thể là student hoặc teacher)
    if hasattr(model, 'semi_test_cfg') and model.semi_test_cfg.get('predict_on') == 'teacher':
        # Sử dụng teacher model
        inference_model = model.teacher if hasattr(model, 'teacher') else model.detector
    else:
        # Sử dụng student model
        inference_model = model.detector if hasattr(model, 'detector') else model
    
    # Thêm cfg nếu chưa có
    if not hasattr(inference_model, 'cfg'):
        inference_model.cfg = model.cfg
    
    # Run inference
    result = inference_detector(inference_model, img_path)
    return result


def predict_and_save(image_path, score_thr=0.3):
    """
    Dự đoán và lưu ảnh kết quả với bounding boxes
    
    Args:
        image_path: Đường dẫn đến ảnh input
        score_thr: Ngưỡng confidence score
        
    Returns:
        Tuple (output_filename, detections_info) hoặc (None, None) nếu có lỗi
    """
    try:
        # Run inference
        result = inference_mixpl(image_path)
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Get predictions
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Filter by score threshold
        valid_idx = scores >= score_thr
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        labels = labels[valid_idx]
        
        # Draw bounding boxes on image
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this class
            color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
            # Convert RGB to BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
            
            # Draw label background
            label_text = f'{CLASS_NAMES[int(label)]}: {score:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color_bgr, -1)
            
            # Draw label text
            cv2.putText(img, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tạo tên file unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"prediction_{timestamp}_{unique_id}.jpg"
        output_path = os.path.join('output', output_filename)
        
        # Lưu ảnh kết quả
        cv2.imwrite(output_path, img)
        
        print(f"Prediction saved to: {output_path}")
        
        # Tạo thông tin detection để trả về
        detections_info = []
        if len(bboxes) > 0:
            print(f"Detected {len(bboxes)} objects with score >= {score_thr}:")
            for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                class_name = CLASS_NAMES[int(label)]
                detection = {
                    'id': i + 1,
                    'class_name': class_name,
                    'confidence': float(score),
                    'bbox': {
                        'x1': float(bbox[0]),
                        'y1': float(bbox[1]),
                        'x2': float(bbox[2]),
                        'y2': float(bbox[3])
                    }
                }
                detections_info.append(detection)
                print(f"  {i+1}. {class_name}: {score:.3f} - bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("No objects detected")
        
        return output_filename, detections_info
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('about.html')


@app.route('/ai')
def ai():
    """Trang AI detection"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để thực hiện prediction"""
    # Kiểm tra key
    key = request.form.get('key', '')
    if key != 'phuc':
        return jsonify({'error': 'Key không đúng! Vui lòng nhập key "phuc"'}), 401

    # Kiểm tra file upload
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file nào được upload'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400

    # Kiểm tra định dạng file
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Định dạng file không được hỗ trợ. Vui lòng upload: {", ".join(allowed_extensions)}'}), 400

    # Tạo thư mục uploads và output nếu chưa có
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Lưu file upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"upload_{timestamp}_{unique_id}_{file.filename}"
    upload_path = os.path.join('uploads', filename)
    file.save(upload_path)
    print(f"Image uploaded to: {upload_path}")

    # Dự đoán
    output_filename, detections_info = predict_and_save(upload_path, score_thr=0.3)
    
    if output_filename:
        # Tạo summary text
        if detections_info and len(detections_info) > 0:
            summary = f"Phát hiện {len(detections_info)} đối tượng:"
            class_count = {}
            for det in detections_info:
                cls = det['class_name']
                class_count[cls] = class_count.get(cls, 0) + 1
            
            summary_parts = [f"{count} {cls}" for cls, count in class_count.items()]
            summary += " " + ", ".join(summary_parts)
        else:
            summary = "Không phát hiện lỗi nào trên mũi khoan."
        
        return jsonify({
            'success': True,
            'original_image': filename,
            'result_image': output_filename,
            'original_url': url_for('uploaded_file', filename=filename),
            'result_url': url_for('output_file', filename=output_filename),
            'classes': CLASS_NAMES,
            'detections': detections_info,
            'summary': summary,
            'total_detections': len(detections_info) if detections_info else 0
        })
    else:
        return jsonify({'error': 'Có lỗi xảy ra trong quá trình dự đoán'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory('uploads', filename)


@app.route('/output/<filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory('output', filename)


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("MixPL Web Application - Drill Bit Defect Detection")
    print("=" * 60)
    
    # Tạo thư mục cần thiết
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Khởi tạo model
    print("\nInitializing MixPL model...")
    if not init_model():
        print("ERROR: Could not initialize model. Please check your paths and model files.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Starting Flask app on port 12345...")
    print("Access the website at: http://localhost:12345")
    print(f"Model can detect these classes: {CLASS_NAMES}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=12347, debug=False)