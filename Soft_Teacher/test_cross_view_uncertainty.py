"""Test cross-view uncertainty computation logic."""
import torch
from mmengine.structures import InstanceData
from collections import defaultdict


def mock_group_predictions(batch_data_samples):
    """Mock grouping - assume 2 groups of 4 views each."""
    groups = {
        'group_0': [0, 1, 2, 3],
        'group_1': [4, 5, 6, 7]
    }
    return groups


def compute_cross_view_uncertainty(batch_data_samples, reg_uncs_list, groups):
    """Simplified version of _compute_cross_view_uncertainty for testing."""
    cross_view_uncs = []
    
    for idx, data_samples in enumerate(batch_data_samples):
        num_boxes = len(data_samples.gt_instances.bboxes)
        if num_boxes == 0:
            cross_view_uncs.append(torch.zeros(0))
            continue
        
        # Find which group this view belongs to
        view_indices = []
        for gid, vindices in groups.items():
            if idx in vindices:
                view_indices = vindices
                break
        
        if len(view_indices) < 2:
            cross_view_uncs.append(torch.ones(num_boxes) * 0.5)
            continue
        
        # Count how many views predict each class
        labels = data_samples.gt_instances.labels
        class_support = torch.zeros(num_boxes)
        
        for box_idx, label in enumerate(labels):
            views_with_class = 0
            for view_idx in view_indices:
                if view_idx == idx:
                    continue
                view_labels = batch_data_samples[view_idx].gt_instances.labels
                if len(view_labels) > 0 and (view_labels == label).any():
                    views_with_class += 1
            
            class_support[box_idx] = views_with_class / max(len(view_indices) - 1, 1)
        
        # Convert support to uncertainty
        uncertainties = 1.0 - class_support
        cross_view_uncs.append(uncertainties)
    
    return cross_view_uncs


def test_cross_view_uncertainty():
    """Test different scenarios."""
    print("=" * 80)
    print("TEST CROSS-VIEW UNCERTAINTY LOGIC")
    print("=" * 80)
    
    # Scenario 1: Strong multi-view agreement (same class in multiple views)
    print("\n[Scenario 1] Strong agreement - Severe_Rust detected by 3/4 views")
    print("-" * 80)
    
    batch_data_samples = []
    reg_uncs_list = []
    
    # Group 0: 4 views
    for i in range(4):
        instances = InstanceData()
        if i < 3:  # Views 0, 1, 2 detect Severe_Rust
            instances.bboxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
            instances.scores = torch.tensor([0.9])
            instances.labels = torch.tensor([3])  # Severe_Rust
            reg_uncs_list.append(torch.tensor([0.02]))
        else:  # View 3 doesn't detect anything
            instances.bboxes = torch.empty((0, 4))
            instances.scores = torch.empty((0,))
            instances.labels = torch.empty((0,), dtype=torch.long)
            reg_uncs_list.append(torch.zeros(0))
        
        instances.gt_instances = instances
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    # Group 1: 4 views (dummy)
    for i in range(4):
        instances = InstanceData()
        instances.bboxes = torch.empty((0, 4))
        instances.scores = torch.empty((0,))
        instances.labels = torch.empty((0,), dtype=torch.long)
        instances.gt_instances = instances
        reg_uncs_list.append(torch.zeros(0))
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    groups = mock_group_predictions(batch_data_samples)
    cv_uncs = compute_cross_view_uncertainty(batch_data_samples, reg_uncs_list, groups)
    
    print("Group 0 (4 views):")
    for i in range(4):
        if len(cv_uncs[i]) > 0:
            print(f"  View {i}: Label={batch_data_samples[i].gt_instances.labels.tolist()}, "
                  f"Support from {torch.sum(batch_data_samples[i].gt_instances.labels == 3).item()}/3 other views, "
                  f"Cross-view uncertainty={cv_uncs[i].item():.3f}")
        else:
            print(f"  View {i}: No predictions")
    
    expected = 2/3  # 2 other views out of 3 have Severe_Rust
    actual = cv_uncs[0].item() if len(cv_uncs[0]) > 0 else None
    assert actual is not None, "View 0 should have uncertainty"
    expected_unc = 1.0 - expected
    print(f"\n✓ Expected uncertainty: {expected_unc:.3f} (support={expected:.3f})")
    print(f"✓ Actual uncertainty: {actual:.3f}")
    assert abs(actual - expected_unc) < 0.01, f"Uncertainty mismatch! Expected {expected_unc:.3f}, got {actual:.3f}"
    
    # Scenario 2: Weak agreement (only 1 view detects)
    print("\n" + "=" * 80)
    print("[Scenario 2] Weak agreement - Chipped detected by only 1/4 views")
    print("-" * 80)
    
    batch_data_samples = []
    reg_uncs_list = []
    
    # Group 0: Only view 0 detects Chipped
    for i in range(4):
        instances = InstanceData()
        if i == 0:
            instances.bboxes = torch.tensor([[30.0, 40.0, 70.0, 80.0]])
            instances.scores = torch.tensor([0.7])
            instances.labels = torch.tensor([1])  # Chipped
            reg_uncs_list.append(torch.tensor([0.05]))
        else:
            instances.bboxes = torch.empty((0, 4))
            instances.scores = torch.empty((0,))
            instances.labels = torch.empty((0,), dtype=torch.long)
            reg_uncs_list.append(torch.zeros(0))
        
        instances.gt_instances = instances
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    # Group 1: dummy
    for i in range(4):
        instances = InstanceData()
        instances.bboxes = torch.empty((0, 4))
        instances.scores = torch.empty((0,))
        instances.labels = torch.empty((0,), dtype=torch.long)
        instances.gt_instances = instances
        reg_uncs_list.append(torch.zeros(0))
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    groups = mock_group_predictions(batch_data_samples)
    cv_uncs = compute_cross_view_uncertainty(batch_data_samples, reg_uncs_list, groups)
    
    print("Group 0 (4 views):")
    for i in range(4):
        if len(cv_uncs[i]) > 0:
            print(f"  View {i}: Label={batch_data_samples[i].gt_instances.labels.tolist()}, "
                  f"Cross-view uncertainty={cv_uncs[i].item():.3f}")
        else:
            print(f"  View {i}: No predictions")
    
    expected_support = 0.0  # 0 other views have Chipped
    expected_unc = 1.0 - expected_support
    actual = cv_uncs[0].item()
    print(f"\n✓ Expected uncertainty: {expected_unc:.3f} (support={expected_support:.3f})")
    print(f"✓ Actual uncertainty: {actual:.3f}")
    assert abs(actual - expected_unc) < 0.01, f"Uncertainty mismatch! Expected {expected_unc:.3f}, got {actual:.3f}"
    
    # Scenario 3: Mixed classes
    print("\n" + "=" * 80)
    print("[Scenario 3] Mixed - View has 2 boxes with different class support")
    print("-" * 80)
    
    batch_data_samples = []
    reg_uncs_list = []
    
    # View 0: 2 boxes (Severe_Rust + Chipped)
    # View 1: 1 box (Severe_Rust only)
    # View 2: 1 box (Severe_Rust only)
    # View 3: nothing
    
    for i in range(4):
        instances = InstanceData()
        if i == 0:
            instances.bboxes = torch.tensor([
                [10.0, 20.0, 50.0, 60.0],  # Severe_Rust
                [100.0, 110.0, 140.0, 150.0]  # Chipped
            ])
            instances.scores = torch.tensor([0.9, 0.7])
            instances.labels = torch.tensor([3, 1])  # Severe_Rust, Chipped
            reg_uncs_list.append(torch.tensor([0.02, 0.05]))
        elif i < 3:  # Views 1, 2
            instances.bboxes = torch.tensor([[15.0, 25.0, 55.0, 65.0]])
            instances.scores = torch.tensor([0.85])
            instances.labels = torch.tensor([3])  # Severe_Rust only
            reg_uncs_list.append(torch.tensor([0.03]))
        else:
            instances.bboxes = torch.empty((0, 4))
            instances.scores = torch.empty((0,))
            instances.labels = torch.empty((0,), dtype=torch.long)
            reg_uncs_list.append(torch.zeros(0))
        
        instances.gt_instances = instances
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    # Group 1: dummy
    for i in range(4):
        instances = InstanceData()
        instances.bboxes = torch.empty((0, 4))
        instances.scores = torch.empty((0,))
        instances.labels = torch.empty((0,), dtype=torch.long)
        instances.gt_instances = instances
        reg_uncs_list.append(torch.zeros(0))
        batch_data_samples.append(type('obj', (object,), {'gt_instances': instances})())
    
    groups = mock_group_predictions(batch_data_samples)
    cv_uncs = compute_cross_view_uncertainty(batch_data_samples, reg_uncs_list, groups)
    
    print("Group 0 (4 views):")
    print("  View 0: 2 boxes")
    print(f"    Box 0 (Severe_Rust): uncertainty={cv_uncs[0][0].item():.3f}")
    print(f"    Box 1 (Chipped): uncertainty={cv_uncs[0][1].item():.3f}")
    
    for i in range(1, 4):
        if len(cv_uncs[i]) > 0:
            print(f"  View {i}: {len(cv_uncs[i])} box(es), uncertainty={cv_uncs[i].tolist()}")
        else:
            print(f"  View {i}: No predictions")
    
    # Box 0 (Severe_Rust): 2 other views have it → support = 2/3 → unc = 1/3
    expected_unc_rust = 1.0 - (2/3)
    actual_unc_rust = cv_uncs[0][0].item()
    print(f"\n✓ Box 0 (Severe_Rust): Expected unc={expected_unc_rust:.3f}, Actual={actual_unc_rust:.3f}")
    assert abs(actual_unc_rust - expected_unc_rust) < 0.01
    
    # Box 1 (Chipped): 0 other views have it → support = 0 → unc = 1.0
    expected_unc_chipped = 1.0
    actual_unc_chipped = cv_uncs[0][1].item()
    print(f"✓ Box 1 (Chipped): Expected unc={expected_unc_chipped:.3f}, Actual={actual_unc_chipped:.3f}")
    assert abs(actual_unc_chipped - expected_unc_chipped) < 0.01
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nKey insights:")
    print("1. Multi-view agreement → LOW uncertainty (confident prediction)")
    print("2. Single-view detection → HIGH uncertainty (likely false positive)")
    print("3. Different classes get different uncertainty scores based on support")
    print("4. This approach works for rotation views without needing spatial alignment!")


if __name__ == '__main__':
    test_cross_view_uncertainty()
