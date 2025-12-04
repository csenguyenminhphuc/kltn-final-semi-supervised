from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, abort
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Cấu hình
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_muikhoan')
ACCESS_KEY = 'phuc'

# Các định dạng file được hỗ trợ
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.avi', '.mov'}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | {'.pdf', '.txt', '.json', '.xml', '.csv'}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext in VIDEO_EXTENSIONS:
        return 'video'
    else:
        return 'file'

def get_folder_contents(folder_path):
    """Lấy nội dung của thư mục"""
    items = []
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            is_dir = os.path.isdir(item_path)
            
            if is_dir:
                # Đếm số file trong thư mục con
                try:
                    count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                except:
                    count = 0
                items.append({
                    'name': item,
                    'type': 'folder',
                    'count': count
                })
            else:
                file_size = os.path.getsize(item_path)
                items.append({
                    'name': item,
                    'type': get_file_type(item),
                    'size': format_size(file_size)
                })
    except Exception as e:
        print(f"Error reading folder: {e}")
    
    # Sắp xếp: thư mục trước, sau đó file
    items.sort(key=lambda x: (0 if x['type'] == 'folder' else 1, x['name'].lower()))
    return items

def format_size(size):
    """Format kích thước file"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

@app.route('/')
def index():
    if session.get('authenticated'):
        return redirect(url_for('browse'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key = request.form.get('key', '')
        if key == ACCESS_KEY:
            session['authenticated'] = True
            return redirect(url_for('browse'))
        else:
            error = 'Key không đúng. Vui lòng thử lại!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/browse')
@app.route('/browse/<path:subpath>')
@login_required
def browse(subpath=''):
    # Xây dựng đường dẫn thực tế
    current_path = os.path.join(DATA_FOLDER, subpath)
    
    # Kiểm tra bảo mật - đảm bảo không truy cập ngoài DATA_FOLDER
    real_path = os.path.realpath(current_path)
    real_data_folder = os.path.realpath(DATA_FOLDER)
    
    if not real_path.startswith(real_data_folder):
        abort(403)
    
    if not os.path.exists(current_path):
        abort(404)
    
    if os.path.isfile(current_path):
        # Nếu là file, trả về file để tải xuống
        directory = os.path.dirname(current_path)
        filename = os.path.basename(current_path)
        return send_from_directory(directory, filename, as_attachment=True)
    
    # Lấy nội dung thư mục
    items = get_folder_contents(current_path)
    
    # Xây dựng breadcrumb
    breadcrumbs = []
    if subpath:
        parts = subpath.split('/')
        for i, part in enumerate(parts):
            breadcrumbs.append({
                'name': part,
                'path': '/'.join(parts[:i+1])
            })
    
    return render_template('browse.html', 
                         items=items, 
                         current_path=subpath,
                         breadcrumbs=breadcrumbs)

@app.route('/view/<path:filepath>')
@login_required
def view_file(filepath):
    """Xem file (ảnh, video)"""
    file_path = os.path.join(DATA_FOLDER, filepath)
    
    # Kiểm tra bảo mật
    real_path = os.path.realpath(file_path)
    real_data_folder = os.path.realpath(DATA_FOLDER)
    
    if not real_path.startswith(real_data_folder):
        abort(403)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        abort(404)
    
    file_type = get_file_type(filepath)
    filename = os.path.basename(filepath)
    
    # Lấy danh sách ảnh trong cùng thư mục để navigation
    parent_dir = os.path.dirname(file_path)
    siblings = []
    current_index = 0
    
    try:
        all_files = sorted([f for f in os.listdir(parent_dir) 
                          if os.path.isfile(os.path.join(parent_dir, f)) 
                          and get_file_type(f) == 'image'])
        parent_subpath = os.path.dirname(filepath)
        
        for i, f in enumerate(all_files):
            if f == filename:
                current_index = i
            siblings.append(os.path.join(parent_subpath, f) if parent_subpath else f)
    except:
        pass
    
    return render_template('view.html',
                         filepath=filepath,
                         filename=filename,
                         file_type=file_type,
                         siblings=siblings,
                         current_index=current_index)

@app.route('/file/<path:filepath>')
@login_required
def serve_file(filepath):
    """Phục vụ file để hiển thị"""
    file_path = os.path.join(DATA_FOLDER, filepath)
    
    # Kiểm tra bảo mật
    real_path = os.path.realpath(file_path)
    real_data_folder = os.path.realpath(DATA_FOLDER)
    
    if not real_path.startswith(real_data_folder):
        abort(403)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        abort(404)
    
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return send_from_directory(directory, filename)

@app.route('/download/<path:filepath>')
@login_required
def download_file(filepath):
    """Tải xuống file"""
    file_path = os.path.join(DATA_FOLDER, filepath)
    
    # Kiểm tra bảo mật
    real_path = os.path.realpath(file_path)
    real_data_folder = os.path.realpath(DATA_FOLDER)
    
    if not real_path.startswith(real_data_folder):
        abort(403)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        abort(404)
    
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=12348)
