import sys
import os
import importlib

# Thêm đường dẫn đến Unbiased_Teacher
sys.path.append('/home/coder/data/trong/KLTN/Unbiased_Teacher')

from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import cv2
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from ubteacher.engine.trainer import CustomPredictor
from ubteacher.config import add_ubteacher_config
import uuid
from datetime import datetime

# BẮT BUỘC: import để đăng ký các lớp của Unbiased Teacher vào registry
importlib.import_module("ubteacher.modeling.meta_arch.rcnn")
importlib.import_module("ubteacher.modeling.roi_heads.roi_heads")
importlib.import_module("ubteacher.modeling.proposal_generator.rpn")

# Set random seed để kết quả nhất quán
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

app = Flask(__name__)


# Khởi tạo model
def init_model():
    try:
        # Đăng ký datasets
        # register_coco_instances("TRAIN_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/train/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/train")
        # register_coco_instances("VAL_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/valid/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/valid")
        register_coco_instances(
            "TRAIN_DATASET",
            {},
            "/home/coder/data/trong/KLTN/data_drill_3/anno_train/_annotations_filtered.coco.json",
            "/home/coder/data/trong/KLTN/data_drill_3/train",
        )
        register_coco_instances(
            "VAL_DATASET",
            {},
            "/home/coder/data/trong/KLTN/data_drill_3/anno_valid/_annotations_filtered.coco.json",
            "/home/coder/data/trong/KLTN/data_drill_3/valid",
        )

        # Cấu hình model
        cfg = get_cfg()
        add_ubteacher_config(cfg)  # BẮT BUỘC: thêm config của Unbiased Teacher
        
        # Dùng đúng file config đã train
        cfg.merge_from_file("/home/coder/data/trong/KLTN/Unbiased_Teacher/configs/Base-RCNN-FPN.yaml")
        
        # Checkpoint đã train xong
        cfg.MODEL.WEIGHTS = "/home/coder/data/trong/KLTN/Unbiased_Teacher/output_40/model_best.pth"
        
        # Số lớp phải khớp lúc train
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        
        # Ngưỡng hiển thị kết quả
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        
        # Kiểm tra GPU
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        
        cfg.DATASETS.TRAIN = ("TRAIN_DATASET", )
        cfg.DATASETS.TEST = ("VAL_DATASET", )
        
        # Freeze config
        cfg.freeze()

        # Metadata với tên class đúng
        meta = MetadataCatalog.get("VAL_DATASET")
        
        # Đặt tên class chính xác theo COCO annotations
        #meta.thing_classes = ['drill', 'Gay', 'Me', 'Mon_dau', 'Ri_set', 'Xuoc']
        meta.thing_classes=["drill",'Broken','Chipped','Scratched','Severe_Rust','Tip_Wear']


        # Đặt bảng màu cố định (RGB 0-255) cho 7 lớp (bao gồm cả "objects")
        meta.thing_colors = [
            (255, 255, 0),
            (134, 34, 255),   
            (0, 255, 206),
            (199, 252, 0), 
            (254, 0, 86), 
            (255, 128, 0), 
            
        ]

        predictor = CustomPredictor(cfg)
        
        # ĐẶT MODEL Ở CHẾ ĐỘ EVALUATION để kết quả nhất quán
        predictor.model.eval()
        
        print("Model initialized successfully!")
        print(f"Class names: {meta.thing_classes}")
        return predictor, meta
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None, None

# Khởi tạo model khi start app
print("Initializing AI model...")
predictor, meta = init_model()

if predictor is None:
    print("ERROR: Could not initialize model. Please check your paths and model files.")
    sys.exit(1)

def predict_and_save(image_path):
    """Dự đoán và lưu ảnh kết quả - sử dụng chính xác thuật toán từ notebook
    
    Returns:
        Tuple (output_filename, detections_info) hoặc (None, None) nếu có lỗi
    """
    try:
        # Đọc ảnh
        im_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if im_bgr is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None
        
        # Nếu ảnh là grayscale 1 kênh -> chuyển sang BGR 3 kênh
        if im_bgr.ndim == 2:
            im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_GRAY2BGR)
        # Nếu ảnh có 4 kênh (RGBA) -> chuyển sang BGR
        elif im_bgr.shape[2] == 4:
            im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_BGRA2BGR)
            
        outputs = predictor(im_bgr)

        # Sử dụng chính xác code từ notebook
        class_names = getattr(meta, "thing_classes", [])
        dataset_id_to_contig = getattr(meta, "thing_dataset_id_to_contiguous_id", None)

        # ===== Prediction: phóng to font bằng scale =====
        im_rgb = im_bgr[:, :, ::-1]
        # Sử dụng ColorMode.SEGMENTATION để màu cố định theo class (từ thing_colors)
        v = Visualizer(im_rgb, metadata=meta, scale=1.6, instance_mode=ColorMode.SEGMENTATION)
        vis_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_rgb = vis_pred.get_image()

        # Tạo tên file unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"prediction_{timestamp}_{unique_id}.jpg"
        output_path = os.path.join('output', output_filename)

        # Lưu ảnh prediction (thay vì plt.show())
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_rgb)  # Visualizer trả RGB
        plt.title("AI Prediction Results", fontsize=16, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1, 
                   facecolor='white', edgecolor='none')
        plt.close()  # Đóng figure để giải phóng memory

        print(f"Prediction saved to: {output_path}")
        
        # Tạo thông tin detection để trả về
        detections_info = []
        inst = outputs["instances"].to("cpu")
        if inst.has("pred_classes") and len(inst) > 0:
            classes_detected = inst.pred_classes.numpy()
            scores = inst.scores.numpy()
            bboxes = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else []
            
            print(f"Detected {len(classes_detected)} objects:")
            for i, (cls_id, score) in enumerate(zip(classes_detected, scores)):
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                detection = {
                    'id': i + 1,
                    'class_name': cls_name,
                    'confidence': float(score),
                    'bbox': {
                        'x1': float(bboxes[i][0]) if len(bboxes) > i else 0,
                        'y1': float(bboxes[i][1]) if len(bboxes) > i else 0,
                        'x2': float(bboxes[i][2]) if len(bboxes) > i else 0,
                        'y2': float(bboxes[i][3]) if len(bboxes) > i else 0
                    }
                }
                detections_info.append(detection)
                print(f"  - {cls_name} (confidence: {score:.3f})")
        else:
            print("No objects detected")
            
        return output_filename, detections_info
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

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
    if key != 'kltn':
        return jsonify({'error': 'Key không đúng! Vui lòng nhập key "kltn"'}), 401

    # Kiểm tra file upload
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file nào được upload'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400

    # Lưu file upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"upload_{timestamp}_{unique_id}_{file.filename}"
    upload_path = os.path.join('uploads', filename)
    file.save(upload_path)
    print(f"Image uploaded to: {upload_path}")

    # Dự đoán
    output_filename, detections_info = predict_and_save(upload_path)
    
    if output_filename:
        # Lấy class names (bỏ class đầu tiên 'drill')
        display_classes = meta.thing_classes[1:] if len(meta.thing_classes) > 1 else meta.thing_classes
        
        # Lọc detections (bỏ class 'drill' nếu có)
        filtered_detections = [d for d in detections_info if d['class_name'] != 'drill'] if detections_info else []
        
        # Tạo summary text
        if filtered_detections and len(filtered_detections) > 0:
            summary = f"Phát hiện {len(filtered_detections)} đối tượng:"
            class_count = {}
            for det in filtered_detections:
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
            'classes': display_classes,
            'detections': filtered_detections,
            'summary': summary,
            'total_detections': len(filtered_detections)
        })
    else:
        return jsonify({'error': 'Có lỗi xảy ra trong quá trình dự đoán'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    print("Starting Flask app on port 12345...")
    print("Access the website at: http://localhost:12345")
    print(f"Model can detect these classes: {meta.thing_classes}")
    app.run(host='0.0.0.0', port=12345, debug=True)