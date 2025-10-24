import sys
import os

# Thêm đường dẫn đến Soft_Teacher và mmdetection
sys.path.append('/home/coder/trong/KLTN_SEMI/code/Soft_Teacher')
sys.path.append('/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/mmdetection')

from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmengine.config import Config
import mmcv
import torch
import uuid
from datetime import datetime

app = Flask(__name__)

# Khởi tạo model theo đúng cách từ notebook
def init_model():
    try:
        # Đăng ký tất cả modules
        register_all_modules()
        
        # Đường dẫn config và checkpoint
        soft_cfg = '/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/work_dirs/soft_teacher_custom_40/20251007_081326/vis_data/config.py'
        ckpt = '/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/work_dirs/soft_teacher_custom_40/epoch_1.pth'
        
        print(f"Loading config from: {soft_cfg}")
        print(f"Loading checkpoint from: {ckpt}")
        
        # 1) Flatten config: SoftTeacher -> detector thuần
        cfg = Config.fromfile(soft_cfg)
        dcfg = cfg.copy()
        dcfg.model = cfg.model.detector
        for k in ['semi_train_cfg', 'semi_test_cfg']:
            if k in dcfg:
                dcfg.pop(k)
        if 'data_preprocessor' in dcfg:
            dcfg.model.setdefault('data_preprocessor', dcfg['data_preprocessor'])
        
        # 2) Init detector *không* load ckpt ở đây
        model = init_detector(dcfg, checkpoint=None, device='cuda:0')
        
        # 3) Đọc ckpt và lọc đúng nhánh (ưu tiên teacher.detector.)
        raw = torch.load(ckpt, map_location='cpu')
        state = raw.get('state_dict', raw)
        
        def strip_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        
        filtered = None
        # Ưu tiên nhánh đầy đủ nhất
        for p in ['teacher.detector.', 'teacher.', 'student.detector.']:
            sub = strip_prefix(state, p)
            if sub:
                filtered = sub
                print(f'Using prefix: {p}  ->  {len(sub)} params')
                break
        
        # Fallback: nếu không có prefix trên, thử trường hợp ckpt đã là detector thuần
        if filtered is None:
            base_ok = {'backbone','neck','rpn_head','roi_head','bbox_head'}
            filtered = {k: v for k, v in state.items() if k.split('.')[0] in base_ok}
            print('Fallback raw detector keys:', len(filtered))
        
        # (Tuỳ) bỏ 'module.' nếu ckpt từng train với DataParallel
        if any(k.startswith('module.') for k in filtered):
            filtered = {k.replace('module.', '', 1): v for k, v in filtered.items()}
        
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print('=> load_state_dict done')
        print('missing:', len(missing), 'unexpected:', len(unexpected))
        
        # Ép model dùng đúng tên lớp
        MY_CLASSES = ('Gay','Me','Mon_dau','Ri_set','Xuoc')
        model.dataset_meta = {'classes': MY_CLASSES}
        
        # Màu sắc cho từng class
        MY_PALETTE = [(255,0,0),(0,0,255),(0,255,0),(255,255,0),(128,0,128)]
        model.dataset_meta['palette'] = MY_PALETTE
        
        # Tạo visualizer
        vis = VISUALIZERS.build(model.cfg.visualizer)
        vis.dataset_meta = model.dataset_meta
        
        print("Soft Teacher model initialized successfully!")
        print(f"Class names: {MY_CLASSES}")
        return model, vis, MY_CLASSES, MY_PALETTE
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Khởi tạo model khi start app
print("Initializing Soft Teacher AI model...")
model, vis, class_names, colors = init_model()

if model is None:
    print("ERROR: Could not initialize model. Please check your paths and model files.")
    sys.exit(1)

def predict_and_save(image_path):
    """Dự đoán và lưu ảnh kết quả sử dụng Soft Teacher model theo đúng cách từ notebook"""
    try:
        # Đọc ảnh
        img = mmcv.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Thực hiện inference
        res = inference_detector(model, img)
        
        # Tạo tên file unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"prediction_st_{timestamp}_{unique_id}.jpg"
        output_path = os.path.join('output', output_filename)
        
        # Sử dụng visualizer để tạo ảnh kết quả (giống như trong notebook)
        vis.add_datasample('res', img, data_sample=res, draw_gt=False, 
                          pred_score_thr=0.5, out_file=output_path)
        
        print(f"Soft Teacher prediction saved to: {output_path}")
        
        # In thông tin detection để debug
        if hasattr(res, 'pred_instances') and len(res.pred_instances) > 0:
            pred_instances = res.pred_instances
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            # Lọc theo threshold
            valid_mask = scores > 0.5
            valid_scores = scores[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) > 0:
                print(f"Detected {len(valid_labels)} objects:")
                for label, score in zip(valid_labels, valid_scores):
                    cls_name = class_names[label] if label < len(class_names) else f"class_{label}"
                    print(f"  - {cls_name} (confidence: {score:.3f})")
            else:
                print("No objects detected above threshold")
        else:
            print("No objects detected")
            
        return output_filename
        
    except Exception as e:
        print(f"Error in Soft Teacher prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/ai')
def ai():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # Lưu file upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"upload_st_{timestamp}_{unique_id}_{file.filename}"
    upload_path = os.path.join('uploads', filename)
    file.save(upload_path)
    print(f"Image uploaded to: {upload_path}")

    # Dự đoán bằng Soft Teacher
    output_filename = predict_and_save(upload_path)
    
    if output_filename:
        return jsonify({
            'success': True,
            'original_image': filename,
            'result_image': output_filename,
            'original_url': url_for('uploaded_file', filename=filename),
            'result_url': url_for('output_file', filename=output_filename),
            'classes': class_names,
            'model_type': 'Soft Teacher'
        })
    else:
        return jsonify({'error': 'Có lỗi xảy ra trong quá trình dự đoán với Soft Teacher'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    print("Starting Soft Teacher Flask app on port 12345...")
    print("Access the website at: http://localhost:12345")
    print(f"Soft Teacher model can detect these classes: {class_names}")
    app.run(host='0.0.0.0', port=12346, debug=True)