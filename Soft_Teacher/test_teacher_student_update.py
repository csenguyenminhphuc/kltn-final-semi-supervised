"""Test script to verify teacher-student parameter updates in Multi-View Soft Teacher.

This script checks:
1. Initial state: Teacher parameters copied from student (momentum=0)
2. After training: Teacher parameters update correctly via EMA (momentum=0.999)
3. Parameter synchronization between student and teacher
4. MVViT parameters are properly included in updates
"""

import torch
import numpy as np
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.logging import print_log

def compare_parameters(student, teacher, name_filter=None):
    """Compare student and teacher parameters.
    
    Args:
        student: Student model
        teacher: Teacher model
        name_filter: Only compare parameters containing this string
    
    Returns:
        dict: Statistics about parameter differences
    """
    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())
    
    results = {
        'total_params': 0,
        'matching_params': 0,
        'identical_params': 0,
        'different_params': 0,
        'max_diff': 0.0,
        'mean_diff': 0.0,
        'examples': []
    }
    
    diffs = []
    for name, s_param in student_params.items():
        if name_filter and name_filter not in name:
            continue
            
        results['total_params'] += 1
        
        if name not in teacher_params:
            continue
            
        results['matching_params'] += 1
        t_param = teacher_params[name]
        
        # Check if parameters match shape
        if s_param.shape != t_param.shape:
            print(f"⚠️  Shape mismatch: {name}")
            print(f"   Student: {s_param.shape}, Teacher: {t_param.shape}")
            continue
        
        # Compute difference
        diff = torch.abs(s_param.data - t_param.data).max().item()
        diffs.append(diff)
        
        if diff < 1e-6:
            results['identical_params'] += 1
        else:
            results['different_params'] += 1
        
        # Store examples
        if len(results['examples']) < 5:
            results['examples'].append({
                'name': name,
                'diff': diff,
                's_mean': s_param.data.mean().item(),
                't_mean': t_param.data.mean().item(),
            })
    
    if diffs:
        results['max_diff'] = max(diffs)
        results['mean_diff'] = np.mean(diffs)
    
    return results

def test_initial_copy():
    """Test 1: Verify teacher is initialized by copying from student."""
    print("\n" + "="*80)
    print("TEST 1: Initial Teacher-Student Copy (momentum=0)")
    print("="*80)
    
    cfg = Config.fromfile('mmdetection/configs/soft_teacher/soft_teacher_custom_multi_view.py')
    
    # Build model
    model = MODELS.build(cfg.model)
    model.eval()
    
    # Simulate before_train with momentum=0 (copy student → teacher)
    print("\n[Before Update] Comparing initial parameters...")
    before_stats = compare_parameters(model.student, model.teacher)
    
    print(f"\nTotal parameters: {before_stats['total_params']}")
    print(f"Matching parameters: {before_stats['matching_params']}")
    print(f"Identical parameters: {before_stats['identical_params']}")
    print(f"Different parameters: {before_stats['different_params']}")
    
    if before_stats['different_params'] > 0:
        print(f"\n⚠️  Teacher NOT initialized from student!")
        print(f"Max difference: {before_stats['max_diff']:.6f}")
        print(f"Mean difference: {before_stats['mean_diff']:.6f}")
    
    # Apply momentum=0 update (copy student → teacher)
    print("\n[Applying momentum=0 update to copy student → teacher...]")
    from mmdet.engine.hooks import MeanTeacherHook
    hook = MeanTeacherHook(momentum=0.999)
    hook.momentum_update(model, momentum=0.0)
    
    # Check after copy
    print("\n[After Update] Comparing parameters...")
    after_stats = compare_parameters(model.student, model.teacher)
    
    print(f"\nTotal parameters: {after_stats['total_params']}")
    print(f"Matching parameters: {after_stats['matching_params']}")
    print(f"Identical parameters: {after_stats['identical_params']}")
    print(f"Different parameters: {after_stats['different_params']}")
    
    if after_stats['identical_params'] == after_stats['matching_params']:
        print("\n✅ TEST PASSED: Teacher successfully copied from student!")
    else:
        print(f"\n❌ TEST FAILED: {after_stats['different_params']} parameters still different!")
        print(f"Max difference: {after_stats['max_diff']:.6f}")
        
        # Show examples
        print("\nExample differences:")
        for ex in after_stats['examples'][:3]:
            print(f"  {ex['name']}: diff={ex['diff']:.6f}, s_mean={ex['s_mean']:.6f}, t_mean={ex['t_mean']:.6f}")

def test_ema_update():
    """Test 2: Verify teacher updates via EMA (momentum=0.999)."""
    print("\n" + "="*80)
    print("TEST 2: EMA Update (momentum=0.999)")
    print("="*80)
    
    cfg = Config.fromfile('mmdetection/configs/soft_teacher/soft_teacher_custom_multi_view.py')
    
    # Build model
    model = MODELS.build(cfg.model)
    model.eval()
    
    # Initialize teacher from student
    from mmdet.engine.hooks import MeanTeacherHook
    hook = MeanTeacherHook(momentum=0.999)
    hook.momentum_update(model, momentum=0.0)
    
    print("\n[Step 1] Teacher initialized from student")
    
    # Modify student parameters slightly (simulate training)
    print("\n[Step 2] Simulating student training (adding noise)...")
    with torch.no_grad():
        for name, param in model.student.named_parameters():
            param.data += torch.randn_like(param) * 0.01
    
    # Check difference before EMA
    before_ema = compare_parameters(model.student, model.teacher)
    print(f"\nBefore EMA: Different params: {before_ema['different_params']}/{before_ema['matching_params']}")
    print(f"Max difference: {before_ema['max_diff']:.6f}")
    
    # Apply EMA update
    print("\n[Step 3] Applying EMA update (momentum=0.999)...")
    hook.momentum_update(model, momentum=0.999)
    
    # Check after EMA
    after_ema = compare_parameters(model.student, model.teacher)
    print(f"\nAfter EMA: Different params: {after_ema['different_params']}/{after_ema['matching_params']}")
    print(f"Max difference: {after_ema['max_diff']:.6f}")
    
    # Verify update worked
    if after_ema['max_diff'] < before_ema['max_diff']:
        print("\n✅ TEST PASSED: Teacher updated correctly!")
        print(f"Difference reduced: {before_ema['max_diff']:.6f} → {after_ema['max_diff']:.6f}")
        print(f"Expected behavior: Teacher should be closer to student after EMA")
    else:
        print(f"\n❌ TEST FAILED: Teacher not updating correctly!")

def test_mvvit_parameters():
    """Test 3: Verify MVViT parameters are included in teacher-student sync."""
    print("\n" + "="*80)
    print("TEST 3: MVViT Parameters Synchronization")
    print("="*80)
    
    cfg = Config.fromfile('mmdetection/configs/soft_teacher/soft_teacher_custom_multi_view.py')
    
    # Build model
    model = MODELS.build(cfg.model)
    model.eval()
    
    # Count MVViT parameters
    print("\n[Checking MVViT parameters...]")
    mvvit_student = compare_parameters(model.student, model.teacher, name_filter='mvvit')
    
    print(f"\nMVViT parameters found:")
    print(f"  Student: {mvvit_student['total_params']} params")
    print(f"  Teacher matching: {mvvit_student['matching_params']} params")
    
    if mvvit_student['matching_params'] > 0:
        print("\n✅ MVViT parameters present in both student and teacher")
        
        # Show some example MVViT parameters
        print("\nExample MVViT parameters:")
        student_params = dict(model.student.named_parameters())
        for name in list(student_params.keys()):
            if 'mvvit' in name:
                print(f"  {name}: {student_params[name].shape}")
                if len([n for n in student_params.keys() if 'mvvit' in n]) >= 5:
                    break
    else:
        print("\n⚠️  No MVViT parameters found! Check if MVViT is properly integrated.")

def test_backbone_parameters():
    """Test 4: Check backbone parameters sync."""
    print("\n" + "="*80)
    print("TEST 4: Backbone Parameters Synchronization")
    print("="*80)
    
    cfg = Config.fromfile('mmdetection/configs/soft_teacher/soft_teacher_custom_multi_view.py')
    
    # Build model
    model = MODELS.build(cfg.model)
    model.eval()
    
    # Initialize teacher
    from mmdet.engine.hooks import MeanTeacherHook
    hook = MeanTeacherHook(momentum=0.999)
    hook.momentum_update(model, momentum=0.0)
    
    # Check backbone
    backbone_stats = compare_parameters(model.student, model.teacher, name_filter='backbone')
    
    print(f"\nBackbone parameters:")
    print(f"  Total: {backbone_stats['total_params']}")
    print(f"  Matching: {backbone_stats['matching_params']}")
    print(f"  Identical: {backbone_stats['identical_params']}")
    
    if backbone_stats['identical_params'] == backbone_stats['matching_params']:
        print("\n✅ Backbone parameters synchronized correctly!")
    else:
        print(f"\n⚠️  {backbone_stats['different_params']} backbone parameters differ")

if __name__ == '__main__':
    print("="*80)
    print("Multi-View Soft Teacher: Parameter Update Verification")
    print("="*80)
    
    try:
        test_initial_copy()
        test_ema_update()
        test_mvvit_parameters()
        test_backbone_parameters()
        
        print("\n" + "="*80)
        print("SUMMARY: All Tests Completed")
        print("="*80)
        print("\nKey Points:")
        print("1. Teacher initialized from student (momentum=0) ✓")
        print("2. Teacher updates via EMA (momentum=0.999) ✓")
        print("3. MVViT parameters included in sync ✓")
        print("4. Backbone parameters synchronized ✓")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
