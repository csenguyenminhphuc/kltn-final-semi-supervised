import sys
import os

# Thêm đường dẫn đến Unbiased_Teacher
sys.path.append('/home/coder/trong/KLTN_SEMI/code/Unbiased_Teacher')

from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from ubteacher.engine.trainer import CustomPredictor
import uuid
from datetime import datetime

app = Flask(__name__)


# Khởi tạo model
def init_model():
    try:
        # Đăng ký datasets
        register_coco_instances("TRAIN_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/train/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/train")
        register_coco_instances("VAL_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/valid/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/valid")

        # Cấu hình model
        cfg = get_cfg()
        cfg.merge_from_file("/home/coder/trong/KLTN_SEMI/code/Unbiased_Teacher/configs/Base-RCNN-FPN.yaml")
        cfg.MODEL.WEIGHTS = os.path.join("/home/coder/trong/temp/model_best.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
        cfg.DATASETS.TRAIN = ("TRAIN_DATASET", )
        cfg.DATASETS.TEST = ("VAL_DATASET", )

        # Metadata với tên class đúng
        meta = MetadataCatalog.get("VAL_DATASET")
        
        # Đặt tên class chính xác theo COCO annotations
        meta.thing_classes = ['drill', 'Gay', 'Me', 'Mon_dau', 'Ri_set', 'Xuoc']
        
        # Đặt bảng màu cố định (RGB 0-255) cho 7 lớp (bao gồm cả "objects")
        meta.thing_colors = [
            (134, 34, 255),   
            (0, 255, 206),  
            (255, 128, 0),   
            (254, 0, 86), 
            (199, 252, 0),  
            (255, 255, 0),
        ]

        predictor = CustomPredictor(cfg)
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
    """Dự đoán và lưu ảnh kết quả - sử dụng chính xác thuật toán từ notebook"""
    try:
        # Đọc ảnh
        im_bgr = cv2.imread(image_path)
        if im_bgr is None:
            print(f"Error: Could not read image from {image_path}")
            return None
            
        outputs = predictor(im_bgr)

        # Sử dụng chính xác code từ notebook
        class_names = getattr(meta, "thing_classes", [])
        dataset_id_to_contig = getattr(meta, "thing_dataset_id_to_contiguous_id", None)

        # ===== Prediction: phóng to font bằng scale =====
        im_rgb = im_bgr[:, :, ::-1]
        v = Visualizer(im_rgb, metadata=meta, scale=1.6)  # <-- chữ & nét to hơn
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
        
        # In thông tin detection để debug
        inst = outputs["instances"].to("cpu")
        if inst.has("pred_classes") and len(inst) > 0:
            classes_detected = inst.pred_classes.numpy()
            scores = inst.scores.numpy()
            print(f"Detected {len(classes_detected)} objects:")
            for i, (cls_id, score) in enumerate(zip(classes_detected, scores)):
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                print(f"  - {cls_name} (confidence: {score:.3f})")
        else:
            print("No objects detected")
            
        return output_filename
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
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
    filename = f"upload_{timestamp}_{unique_id}_{file.filename}"
    upload_path = os.path.join('uploads', filename)
    file.save(upload_path)
    print(f"Image uploaded to: {upload_path}")

    # Dự đoán
    output_filename = predict_and_save(upload_path)
    
    if output_filename:
        return jsonify({
            'success': True,
            'original_image': filename,
            'result_image': output_filename,
            'original_url': url_for('uploaded_file', filename=filename),
            'result_url': url_for('output_file', filename=output_filename),
            'classes': meta.thing_classes[1:]  # Loại bỏ "objects" class
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