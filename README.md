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
- **Deployment**: Docker, Kubernetes

### Development Tools
- **Environment**: Conda, Python 3.9+
- **Version Control**: Git/GitHub
- **DevOps**: CI/CD Pipeline
- **Cloud**: AWS, Cloudflare

## 👥 Đội ngũ phát triển

### 🔧 Nguyễn Minh Phúc - DevSecOps AI Engineer
**Vai trò**: Machine Learning Engineer & DevOps Specialist

**Chuyên môn**:
- 🤖 **AI/ML**: PyTorch, TensorFlow, Scikit-learn, YOLO, CNN
- ⚙️ **DevOps**: Docker, Kubernetes, CI/CD, Linux, AWS
- 🔒 **Security**: DevSecOps, Infrastructure Security
- 💻 **Programming**: Python, JavaScript, Java, C++, SQL

**Trách nhiệm**:
- Thiết kế và phát triển mô hình Semi-supervised Learning
- Tối ưu hóa hiệu suất mô hình AI và pipeline ML
- Xây dựng hạ tầng DevOps và đảm bảo bảo mật hệ thống
- Triển khai tự động và quản lý production environment

### 📊 Phạm Gia Khánh - AI Data Engineer
**Vai trò**: Data Scientist & Full-stack Developer

**Chuyên môn**:
- 📈 **Data Science**: Pandas, NumPy, Matplotlib, Seaborn, Statistics
- 🌐 **Web Development**: React, Node.js, Flask, HTML/CSS
- 🗄️ **Database**: MongoDB, PostgreSQL, Data Warehousing
- 🔧 **Tools**: Anaconda, VS Code, Postman, Firebase, REST API

**Trách nhiệm**:
- Xử lý và phân tích dữ liệu huấn luyện
- Thiết kế và phát triển giao diện web application
- Xây dựng API và quản lý cơ sở dữ liệu
- Chuyển đổi insights từ dữ liệu thành sản phẩm thực tế

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

- **Độ chính xác (mAP)**: 85%+
- **Thời gian xử lý**: ~2-3 giây/ảnh
- **Số lớp phát hiện**: 6 loại lỗi
- **Threshold confidence**: 0.50

### Các lỗi có thể phát hiện:
- 🔩 **Gay**: Lỗi gãy mũi khoan
- 🔴 **Me**: Lỗi mẻ cạnh
- 🟡 **Mon_dau**: Lỗi mòn đầu khoan  
- 🔵 **Ri_set**: Lỗi rạn nứt
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

## 📝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp cho dự án! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu. Vui lòng liên hệ tác giả để biết thêm thông tin về việc sử dụng thương mại.

## 📞 Liên hệ

- **Nguyễn Minh Phúc**: [GitHub](https://github.com/phuc-profile) | Email: phuc@university.edu.vn
- **Phạm Gia Khánh**: [GitHub](https://github.com/khanh-profile) | Email: khanh@university.edu.vn

## 🙏 Acknowledgments

- Khoa Khoa Học Máy Tính - Đại học ABC
- Framework Detectron2 by Facebook AI Research
- Semi-supervised Learning Community
- All contributors and supporters

---

**🎓 Đề tài Khóa luận Tốt nghiệp - Khoa Khoa Học Máy Tính - 2025**

*"Ứng dụng công nghệ AI để giải quyết bài toán thực tế trong sản xuất công nghiệp"*