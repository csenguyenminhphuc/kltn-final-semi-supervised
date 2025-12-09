# Hệ thống Mô hình Học Sâu Bán Giám Sát Phát hiện Lỗi Mũi Khoan từ Ảnh Đa Góc Nhìn

![System Banner](https://img.shields.io/badge/AI-Semi--Supervised%20Learning-blue) ![Framework](https://img.shields.io/badge/Framework-Detectron2-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

## 📋 Tổng quan hệ thống

Hệ thống này được phát triển để tự động phát hiện và phân loại các loại lỗi trên mũi khoan trong môi trường sản xuất công nghiệp sử dụng công nghệ học sâu bán giám sát (Semi-supervised Deep Learning). Với khả năng xử lý ảnh đa góc nhìn, hệ thống đạt độ chính xác cao và có thể triển khai trong thực tế.

![Demo hệ thống](web/info.gif)

### 🎯 Mục tiêu dự án
- Phát hiện tự động các lỗi mũi khoan: **Gay**, **Me**, **Mon_dau**, **Ri_set**, **Xuoc**
- Tối ưu quy trình kiểm tra chất lượng trong sản xuất
- Giảm thiểu sai sót do con người và tăng hiệu quả sản xuất
- Ứng dụng công nghệ AI tiên tiến vào thực tiễn công nghiệp

## 📦 Dataset

### Drill Bit Dataset
Dataset ảnh mũi khoan được sử dụng để huấn luyện và đánh giá mô hình trong dự án này.

🔗 **Link Dataset**: [Kaggle - Drill Bit Dataset](https://www.kaggle.com/datasets/csenguyenminhphuc/drill-bit-dataset)

**Thông tin dataset**:
- **Số lượng ảnh**: Đa dạng ảnh mũi khoan từ nhiều góc nhìn
- **Định dạng**: COCO format annotations
- **Số lớp**: 5 loại lỗi (Gay, Me, Mon_dau, Ri_set, Xuoc)
- **Chia tập**: Train / Validation / Test

## 🖥️ Giám sát hệ thống

Công cụ giám sát server và website hệ thống:

🔗 **Server Monitoring Suite Agent**: [GitHub Repository](https://github.com/csenguyenminhphuc/ServerMonitoringSuite-Agent)

**Tính năng**:
- Giám sát trạng thái server real-time
- Theo dõi tài nguyên hệ thống (CPU, RAM, Disk)
- Cảnh báo khi có sự cố
- Dashboard trực quan

## 🏗️ Kiến trúc hệ thống

```
KLTN/
├── 🤖 Soft_Teacher/              # Framework Soft Teacher (Multi-view)
│   ├── mmdetection/              # MMDetection framework
│   ├── mmengine/                 # MMEngine core
│   ├── tools/                    # Training & inference tools
│   ├── work_dirs/                # Trained models & logs
│   └── train_v3_20.ipynb         # Training notebook
├── 🤖 Soft_Teacher_SingleView/   # Soft Teacher Single View version
├── 🤖 Unbiased_Teacher/          # Thuật toán Unbiased Teacher
│   ├── configs/                  # Configuration files
│   ├── ubteacher/                # Core module
│   ├── output/                   # Training outputs
│   └── inferences/               # Inference results
├── 🤖 Semi-DETR/                 # Semi-supervised DETR
│   ├── configs/                  # Configuration files
│   ├── detr_od/                  # Object detection module
│   ├── detr_ssod/                # Semi-supervised module
│   └── tools/                    # Training & utility tools
├── 🤖 DETR_Mixup/                # DETR với MixPL augmentation
│   ├── MixPL/                    # MixPL module
│   ├── mmdetection/              # MMDetection framework
│   └── train.ipynb               # Training notebook
├── 🔧 PreProcessing/             # Tiền xử lý dữ liệu
│   ├── Instance_Segmentation_Yolov8.ipynb  # Segmentation notebook
│   └── convertSegmentToBBoxVer2.ipynb      # Convert segment to bbox
├── 🔧 yolov11n_train_head_drill/ # YOLOv11 training cho head drill
│   └── train_yolo_head_drill.ipynb         # Training notebook
├── 📊 data_drill/                # Dataset mũi khoan v1
│   ├── train/                    # Training images
│   ├── valid/                    # Validation images
│   ├── anno_train/               # Training annotations
│   ├── anno_valid/               # Validation annotations
│   ├── semi_anns/                # Semi-supervised annotations
│   └── semi_anno_multiview/      # Multi-view annotations
├── 📊 data_drill_2/              # Dataset mũi khoan v2
├── 📊 data_drill_3/              # Dataset mũi khoan v3 (có test set)
├── 🌐 web/                       # Web application chính
│   ├── app.py                    # Flask backend
│   ├── static/                   # CSS, JS, assets
│   ├── templates/                # HTML templates
│   └── info.gif                  # Demo animation
├── 🌐 data_web/                  # Web application phụ
├── 📖 document/                  # Tài liệu và hướng dẫn
├── 📄 analysis_anno.py           # Script phân tích annotations
├── 📄 filter_annotations.py      # Script lọc annotations
└── 📄 inference.py               # Script inference
```

## 🚀 Công nghệ sử dụng

### Backend AI Engine
- **Framework**: Detectron2, MMDetection, PyTorch
- **Architecture**: Faster R-CNN với FPN backbone, DETR Transformer
- **Methods**: 
  - 🔹 **Unbiased Teacher** - Detectron2-based semi-supervised learning
  - 🔹 **Soft Teacher** - MMDetection-based với multi-view support
  - 🔹 **Semi-DETR** - Transformer-based semi-supervised detection
  - 🔹 **MixPL** - Mix Pseudo Labels augmentation
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas

### Web Application
- **Backend**: Flask Framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Docker

### Development Tools
- **Environment**: Conda, Python 3.9+
- **Version Control**: Git/GitHub
- **DevOps**: CI/CD Pipeline
- **Cloud**: Cloudflare

## 👥 Đội ngũ phát triển

### 🔧 Nguyễn Minh Phúc - DevSecOps & Infrastructure Engineer
**Vai trò**: Machine Learning Engineer & DevOps Specialist

**Chuyên môn**:
- 🤖 **AI/ML**: PyTorch, TensorFlow, Scikit-learn
- ⚙️ **DevOps**: Docker, Linux
- 🔒 **Security**: DevSecOps, Infrastructure Security
- 💻 **Programming**: Python, JavaScript

**Trách nhiệm**:
- Thiết kế và phát triển hệ thống sử dụng được mô hình Semi-supervised Learning
- Thiết kế và phát triển giao diện website application
- Xây dựng hạ tầng đảm bảo bảo mật hệ thống
- Tham gia vào quá trình gán nhãn dữ liệu
- Cấu hình server truy cập an toàn, cài đặt các môi trường đảm bảo cho việc huấn luyện mô hình

### 📊 Phạm Gia Khánh - AI Engineer
**Vai trò**: Data Engineer & AI Engineer

**Chuyên môn**:
- 🤖 **AI/ML**: PyTorch, TensorFlow, Scikit-learn
- 📈 **Data Science**: Pandas, NumPy, Matplotlib, Seaborn, Statistics
- 🌐 **Web Development**: Flask, HTML/CSS
- 🔧 **Tools**: Anaconda, VS Code, Postman

**Trách nhiệm**:
- Xử lý và phân tích dữ liệu huấn luyện
- Thiết kế và xây dựng được mô hình Semi-supervised Learning
- Phân tích và trực quan hóa dữ liệu
- Tham gia vào quá trình gán nhãn dữ liệu, chia tập dữ liệu
- Huấn luyện mô hình và đưa ra giải pháp tối ưu cho mô hình

## 📦 Cài đặt và triển khai

### Yêu cầu hệ thống
```bash
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- RAM: 32GB+ recommended
- Storage: 150GB+ available space
- GPU: NVIDIA với ít nhất 32GB VRAM
```

### 1. Clone repository
```bash
git clone https://github.com/csenguyenminhphuc/kltn-final-semi-supervised.git
cd KLTN
```

### 2. Thiết lập môi trường cho Web Application
```bash
# Tạo conda environment cho web
conda create --prefix ./web/.envweb python=3.9.19 -y
conda activate ./web/.envweb

# Cài đặt PyTorch với CUDA
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt Detectron2 (cho Unbiased Teacher)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Cài đặt dependencies web
cd web
pip install -r requirements.txt
```

### 3. Thiết lập môi trường cho Soft Teacher / Semi-DETR
```bash
# Tạo conda environment riêng
conda create -n soft_teacher python=3.9 -y
conda activate soft_teacher

# Cài đặt PyTorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt MMDetection và MMEngine
pip install mmcv-full mmdet mmengine

# Cài đặt các dependencies khác
pip install wandb prettytable opencv-python
```

### 4. Chạy Web Application
```bash
cd web
python app.py
# Hoặc sử dụng gunicorn cho production
gunicorn -c gunicorn_config.py app:app
```

Truy cập hệ thống tại: `http://localhost:12345`
Truy cập hệ thống công khai: `kltn.csenguyenminhphuc.id.vn`

## 🎯 Sử dụng hệ thống

### Web Interface
1. **Truy cập trang chủ**: Tìm hiểu về dự án và đội ngũ
2. **Sử dụng AI**: Upload ảnh mũi khoan để phân tích
3. **Nhập key xác thực**: `phuc` (demo key)
4. **Xem kết quả**: Hệ thống sẽ hiển thị ảnh gốc và kết quả phát hiện lỗi

### API Endpoints
```python
POST /predict          # Dự đoán lỗi từ ảnh upload
GET  /uploads/<file>    # Truy cập ảnh đã upload  
GET  /output/<file>     # Truy cập kết quả dự đoán
```
## 📊 Hiệu suất mô hình

### Kết quả đánh giá tại Iteration 60,000

| Phương pháp | 10% Labeled | 10% Labeled | 10% Labeled | 20% Labeled | 20% Labeled | 20% Labeled | 40% Labeled | 40% Labeled | 40% Labeled |
|:------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| **Metrics** | **mAP** | **mAP50** | **mAP75** | **mAP** | **mAP50** | **mAP75** | **mAP** | **mAP50** | **mAP75** |
| Supervised | 18.51 | 46.76 | 11.49 | 21.23 | 50.47 | 12.40 | 26.00 | 59.78 | 16.82 |
| Unbiased Teacher | 24.2 | 63.1 | 13.1 | 27.4 | 67.7 | 14.6 | 30.4 | 72.1 | 19.1 |
| Soft Teacher | 16.5 | 40.8 | 11.0 | 20.8 | 51.6 | 11.6 | 25.8 | 62.1 | 15.1 |
| **MixPL** | 33.4 | **68.7** | 27.8 | 36.3 | **71.6** | 31.8 | 40.2 | **76.0** | 37.8 |
| Multi View với Soft Teacher | 20.6 | 55.8 | 15.1 | 23.4 | 65.7 | 18.6 | 30.8 | 73.1 | 20.1 |

> 📌 **Ghi chú**: 
> - **mAP**: Mean Average Precision (độ chính xác trung bình)
> - **mAP50**: mAP tại IoU threshold 0.5
> - **mAP75**: mAP tại IoU threshold 0.75
> - Giá trị **in đậm** là kết quả tốt nhất trong từng cột

### Nhận xét kết quả
- **MixPL** đạt hiệu suất cao nhất ở tất cả các tỷ lệ dữ liệu có nhãn (10%, 20%, 40%)
- Với 40% labeled data, MixPL đạt **mAP50 = 76.0%**, vượt trội so với các phương pháp khác
- **Unbiased Teacher** cho kết quả tốt thứ hai, đặc biệt hiệu quả với lượng dữ liệu có nhãn thấp

### Các lỗi có thể phát hiện:
- 🔩 **Gay**: Lỗi gãy mũi khoan
- 🔴 **Me**: Lỗi mẻ mũi khoan
- 🟡 **Mon_dau**: Lỗi mòn đầu khoan  
- 🔵 **Ri_set**: Lỗi rỉ sét
- 🟣 **Xuoc**: Lỗi xước

## 🛠️ Development

### Training Unbiased Teacher
```bash
cd Unbiased_Teacher
python train_net.py --num-gpus 1 --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1_custom.yaml
```

### Training Soft Teacher (MMDetection)
```bash
cd Soft_Teacher
python tools/train.py configs/soft_teacher/soft_teacher_faster_rcnn_r50_fpn.py
```

### Training Semi-DETR
```bash
cd Semi-DETR
python tools/train.py configs/semi_detr/semi_detr_r50.py
```

### Training với MixPL
```bash
cd DETR_Mixup
# Xem notebook train.ipynb để biết chi tiết
```

### Đánh giá mô hình
```bash
# Unbiased Teacher
python inference.py --model-path output/model_best.pth --test-data data_drill_3/test/

# Soft Teacher
python tools/test.py configs/soft_teacher.py work_dirs/latest.pth
```

## 📄 License

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu. Vui lòng liên hệ tác giả để biết thêm thông tin về việc sử dụng thương mại.

## 📞 Liên hệ

- **Nguyễn Minh Phúc**: [GitHub](https://github.com/csenguyenminhphuc) | Email: 22637001.phuc@student.iuh.edu.vn 
- **Phạm Gia Khánh**: [GitHub](https://github.com/cs-khanh) | Email: 22724051.khanh@student.iuh.edu.vn

## 🙏 Acknowledgments
![IUH LOGO](https://iuh.edu.vn/assets/images/icons/logo.svg?v=51)
- Khoa Học Máy Tính - Khoa Công Nghệ Thông Tin - Đại Học Công Nghiệp Thành Phố Hồ Chí Minh 
- Framework Detectron2 by Facebook AI Research
- Framework MMDetection, MMEngine by OpenMMLab
- Semi-supervised Learning Community
- All contributors and supporters

---

**🎓 Đề tài Khóa luận Tốt nghiệp - Khoa Khoa Học Máy Tính - 2025**

*"Ứng dụng công nghệ AI để giải quyết bài toán thực tế trong sản xuất công nghiệp"*