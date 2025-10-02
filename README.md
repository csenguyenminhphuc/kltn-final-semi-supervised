# Hệ thống Mô hình Học Sâu Bán Giám Sát Phát hiện Lỗi Mũi Khoan từ Ảnh Đa Góc Nhìn

![System Banner](https://img.shields.io/badge/AI-Semi--Supervised%20Learning-blue) ![Framework](https://img.shields.io/badge/Framework-Detectron2-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

## 📋 Tổng quan hệ thống

Hệ thống này được phát triển để tự động phát hiện và phân loại các loại lỗi trên mũi khoan trong môi trường sản xuất công nghiệp sử dụng công nghệ học sâu bán giám sát (Semi-supervised Deep Learning). Với khả năng xử lý ảnh đa góc nhìn, hệ thống đạt độ chính xác cao và có thể triển khai trong thực tế.

### 🎯 Mục tiêu dự án
- Phát hiện tự động các lỗi mũi khoan: **Gay**, **Me**, **Mon_dau**, **Ri_set**, **Xuoc_than**
- Tối ưu quy trình kiểm tra chất lượng trong sản xuất
- Giảm thiểu sai sót do con người và tăng hiệu quả sản xuất
- Ứng dụng công nghệ AI tiên tiến vào thực tiễn công nghiệp

## 🏗️ Kiến trúc hệ thống

```
KLTN_SEMI/
├── 🤖 code/                    # Mã nguồn mô hình AI
│   ├── Soft_Teacher/          # Framework Semi-supervised Learning
│   └── Unbiased_Teacher/      # Thuật toán Unbiased Teacher
├── 📊 data/                   # Dữ liệu huấn luyện và kiểm thử
│   ├── train/                 # Dataset training
│   ├── valid/                 # Dataset validation
│   └── test/                  # Dataset testing
├── 🌐 web/                    # Ứng dụng web interface
│   ├── app.py                 # Flask backend
│   ├── static/                # CSS, JS, assets
│   ├── templates/             # HTML templates
│   ├── uploads/               # Thư mục upload ảnh
│   └── output/                # Kết quả dự đoán
├── 📖 document/               # Tài liệu và hướng dẫn
├── 📈 visualize/              # Công cụ trực quan hóa
└── 🔧 output/                 # Kết quả và metrics
```

## 🚀 Công nghệ sử dụng

### Backend AI Engine
- **Framework**: Detectron2, PyTorch
- **Architecture**: Faster R-CNN với FPN backbone
- **Method**: Semi-supervised Learning (Unbiased Teacher)
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

### 🔧 Nguyễn Minh Phúc - DevSecOps AI Engineer
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

### 📊 Phạm Gia Khánh - AI Data Engineer
**Vai trò**: Data Engineer & Machine Learning Engineer

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
- RAM: 16GB+ recommended
- Storage: 50GB+ available space
```

### 1. Clone repository
```bash
git clone https://github.com/your-repo/kltn-final-semi-supervised.git
cd KLTN_SEMI
```

### 2. Thiết lập môi trường
```bash
# Tạo conda environment
conda create --prefix ./web/.envweb python=3.9.19 -y
conda activate ./web/.envweb

# Cài đặt PyTorch với CUDA
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Cài đặt dependencies
```bash
cd web
pip install -r requirements.txt
```

### 4. Chạy ứng dụng
```bash
python app.py
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

- **Độ chính xác (mAP)**: Đang hoàn thiện....
- **Thời gian xử lý**: Đang hoàn thiện....
- **Số lớp phát hiện**: 6 loại lỗi
- **Threshold confidence**: Đang hoàn thiện....

### Các lỗi có thể phát hiện:
- 🔩 **Gay**: Lỗi gãy mũi khoan
- 🔴 **Me**: Lỗi mẻ mũi khoan
- 🟡 **Mon_dau**: Lỗi mòn đầu khoan  
- 🔵 **Ri_set**: Lỗi rỉ sét
- 🟣 **Xuoc_than**: Lỗi xước thân

## 🛠️ Development

### Training mô hình mới
```bash
cd code/Unbiased_Teacher
python train_net.py --num-gpus 1 --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1_custom.yaml
```

### Đánh giá mô hình
```bash
python inference.py --model-path temp/model_best.pth --test-data data/test/
```

## 📄 License

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu. Vui lòng liên hệ tác giả để biết thêm thông tin về việc sử dụng thương mại.

## 📞 Liên hệ

- **Nguyễn Minh Phúc**: [GitHub](https://github.com/csenguyenminhphuc) | Email: 22637001.phuc@student.iuh.edu.vn 
- **Phạm Gia Khánh**: [GitHub](https://github.com/khanhcs) | Email: 22724051.khanh@student.iuh.edu.vn

## 🙏 Acknowledgments
![IUH LOGO](https://iuh.edu.vn/templates/2015/image/logo.png)
- Khoa Học Máy Tính - Khoa Công Nghệ Thông Tin - Đại Học Công Nghiệp Thành Phố Hồ Chí Minh 
- Framework Detectron2 by Facebook AI Research
- Semi-supervised Learning Community
- All contributors and supporters

---

**🎓 Đề tài Khóa luận Tốt nghiệp - Khoa Khoa Học Máy Tính - 2025**

*"Ứng dụng công nghệ AI để giải quyết bài toán thực tế trong sản xuất công nghiệp"*