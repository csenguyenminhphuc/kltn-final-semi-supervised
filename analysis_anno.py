import json
import os
from collections import defaultdict
from tabulate import tabulate

def analyze_annotation_file(file_path):
    """Phân tích một file annotation COCO format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Tạo mapping category_id -> category_name
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Đếm annotations theo category
    category_counts = defaultdict(int)
    total_annotations = len(data['annotations'])
    
    for ann in data['annotations']:
        category_counts[ann['category_id']] += 1
    
    # Tổng số images
    total_images = len(data['images'])
    
    return {
        'total_images': total_images,
        'total_annotations': total_annotations,
        'category_counts': category_counts,
        'categories': categories
    }

def analyze_semi_supervised_data(base_path):
    """Phân tích toàn bộ dữ liệu semi-supervised"""
    
    # Danh sách các file cần phân tích
    configs = [
        # ('1@1.0', '1%'),
        # ('2@1.0', '1%'),
        # ('3@1.0', '1%')
        # ('1@5.0', '5%'),
        # ('2@5.0', '5%'),
        # ('3@5.0', '5%')
        ('1@10.0', '10%'),
        ('2@10.0', '10%'),
        ('3@10.0', '10%')
        # ('1@20.0', '20%'),
        # ('2@20.0', '20%'),
        # ('3@20.0', '20%'),
        # ('1@40.0', '40%'),
        # ('2@40.0', '40%'),
        # ('3@40.0', '40%')
    ]
    
    results = []
    
    for config, percent in configs:
        labeled_file = os.path.join(base_path, f'instances_train.{config}.json')
        unlabeled_file = os.path.join(base_path, f'instances_train.{config}-unlabeled.json')
        
        if not os.path.exists(labeled_file) or not os.path.exists(unlabeled_file):
            continue
        
        print(f"\n{'='*80}")
        print(f"Phân tích: Run {config.split('@')[0]} - {percent} Data")
        print(f"{'='*80}")
        
        # Phân tích labeled data
        labeled_stats = analyze_annotation_file(labeled_file)
        
        # Phân tích unlabeled data
        unlabeled_stats = analyze_annotation_file(unlabeled_file)
        
        # In thống kê tổng quan
        print(f"\n📊 TỔNG QUAN:")
        print(f"   Labeled   : {labeled_stats['total_images']:4d} images, {labeled_stats['total_annotations']:5d} bboxes")
        print(f"   Unlabeled : {unlabeled_stats['total_images']:4d} images, {unlabeled_stats['total_annotations']:5d} bboxes")
        print(f"   Total     : {labeled_stats['total_images'] + unlabeled_stats['total_images']:4d} images, {labeled_stats['total_annotations'] + unlabeled_stats['total_annotations']:5d} bboxes")
        
        # Tạo bảng thống kê theo class
        categories = labeled_stats['categories']
        table_data = []
        
        print(f"\n📈 THỐNG KÊ THEO CLASS:")
        for cat_id, cat_name in sorted(categories.items(), key=lambda x: x[0]):
            labeled_count = labeled_stats['category_counts'][cat_id]
            unlabeled_count = unlabeled_stats['category_counts'][cat_id]
            total_count = labeled_count + unlabeled_count
            
            labeled_percent = (labeled_count / labeled_stats['total_annotations'] * 100) if labeled_stats['total_annotations'] > 0 else 0
            unlabeled_percent = (unlabeled_count / unlabeled_stats['total_annotations'] * 100) if unlabeled_stats['total_annotations'] > 0 else 0
            total_percent = (total_count / (labeled_stats['total_annotations'] + unlabeled_stats['total_annotations']) * 100) if (labeled_stats['total_annotations'] + unlabeled_stats['total_annotations']) > 0 else 0
            
            table_data.append([
                cat_name,
                f"{labeled_count:5d} ({labeled_percent:5.2f}%)",
                f"{unlabeled_count:5d} ({unlabeled_percent:5.2f}%)",
                f"{total_count:5d} ({total_percent:5.2f}%)"
            ])
        
        # Thêm dòng tổng
        table_data.append([
            "TOTAL",
            f"{labeled_stats['total_annotations']:5d} (100.00%)",
            f"{unlabeled_stats['total_annotations']:5d} (100.00%)",
            f"{labeled_stats['total_annotations'] + unlabeled_stats['total_annotations']:5d} (100.00%)"
        ])
        
        print(tabulate(
            table_data,
            headers=["Class", "Labeled", "Unlabeled", "Total"],
            tablefmt="grid"
        ))
        
        # Tính tỷ lệ labeled/unlabeled
        ratio = labeled_stats['total_annotations'] / unlabeled_stats['total_annotations'] if unlabeled_stats['total_annotations'] > 0 else 0
        print(f"\n📊 TỶ LỆ: Labeled/Unlabeled = {ratio:.4f} (1:{1/ratio:.2f})")
        
        # Lưu kết quả để so sánh
        results.append({
            'config': config,
            'percent': percent,
            'labeled': labeled_stats,
            'unlabeled': unlabeled_stats
        })
    
    # So sánh giữa các % data
    print(f"\n{'='*80}")
    print(f"SO SÁNH GIỮA CÁC % DATA")
    print(f"{'='*80}")
    
    # Nhóm theo run
    runs = {}
    for result in results:
        run = result['config'].split('@')[0]
        if run not in runs:
            runs[run] = []
        runs[run].append(result)
    
    for run, run_results in sorted(runs.items()):
        print(f"\n🔍 Run {run}:")
        comparison_data = []
        
        for result in sorted(run_results, key=lambda x: float(x['percent'].strip('%'))):
            percent = result['percent']
            labeled = result['labeled']['total_annotations']
            unlabeled = result['unlabeled']['total_annotations']
            total = labeled + unlabeled
            
            comparison_data.append([
                percent,
                f"{labeled:5d}",
                f"{unlabeled:5d}",
                f"{total:5d}",
                f"{labeled/total*100:.2f}%"
            ])
        
        print(tabulate(
            comparison_data,
            headers=["% Data", "Labeled", "Unlabeled", "Total", "% Labeled"],
            tablefmt="grid"
        ))

if __name__ == "__main__":
    # Đường dẫn tới thư mục chứa annotations
    base_path = "/home/coder/data/trong/KLTN/data_drill_3/semi_anns"
    
    print("🔍 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU SEMI-SUPERVISED")
    print("="*80)
    
    analyze_semi_supervised_data(base_path)
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH PHÂN TÍCH")
    print(f"{'='*80}")
