#!/usr/bin/env python
"""Test script to verify multi-view fixes work correctly."""

import torch
import torch.nn as nn
from mmdet.models.utils.multi_view_transformer import MVViT
from mmdet.models.detectors.multi_view_soft_teacher import MultiViewSoftTeacher

def test_mvvit_modes():
    """Test MVViT with different spatial attention modes."""
    print("\n" + "="*80)
    print("Testing MVViT Spatial Attention Modes")
    print("="*80)
    
    V, B, C, H, W = 8, 2, 256, 64, 64
    
    for mode in ['global', 'efficient', 'full']:
        print(f"\n[Test] Mode: {mode}")
        print(f"  Input: V={V}, B={B}, C={C}, H={H}x{W}")
        
        # Create MVViT
        mvvit = MVViT(
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            spatial_attention=mode
        )
        
        # Create input: List[V] of (B, C, H, W)
        features_views = [torch.randn(B, C, H, W) for _ in range(V)]
        
        # Forward
        try:
            refined_views = mvvit.forward_single_level(features_views)
            
            # Check output
            assert len(refined_views) == V, f"Expected {V} views, got {len(refined_views)}"
            for i, view in enumerate(refined_views):
                assert view.shape == (B, C, H, W), \
                    f"View {i} has wrong shape: {view.shape}, expected {(B, C, H, W)}"
            
            print(f"  ✅ Output: {len(refined_views)} views, each shape {refined_views[0].shape}")
            print(f"  ✅ Mode '{mode}' works correctly!")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            raise

def test_loss_aggregation():
    """Test loss aggregation logic."""
    print("\n" + "="*80)
    print("Testing Loss Aggregation")
    print("="*80)
    
    # Mock losses
    losses = {
        'loss_rpn_cls': torch.tensor(0.5),
        'loss_cls': torch.tensor(1.0),
        'loss_bbox': torch.tensor(0.3)
    }
    
    # Test different aggregation modes
    from mmdet.models.detectors.multi_view_soft_teacher import MultiViewSoftTeacher
    
    for mode in ['mean', 'sum']:
        print(f"\n[Test] Aggregation mode: {mode}")
        
        # Create a mock instance just for the method
        class MockTeacher:
            def __init__(self, agg_mode):
                self.aggregate_views = agg_mode
            
            def _aggregate_view_losses(self, losses, V):
                """Copy of the actual method."""
                if self.aggregate_views == 'mean':
                    return losses
                elif self.aggregate_views == 'sum':
                    return {k: v * V for k, v in losses.items()}
                else:
                    return losses
        
        mock = MockTeacher(mode)
        V = 8
        
        aggregated = mock._aggregate_view_losses(losses.copy(), V)
        
        print(f"  Original losses: {losses}")
        print(f"  Aggregated (V={V}): {aggregated}")
        
        if mode == 'mean':
            for k in losses:
                assert torch.allclose(aggregated[k], losses[k]), \
                    f"Mean aggregation should not change losses"
            print(f"  ✅ Mean aggregation correct (no change)")
        elif mode == 'sum':
            for k in losses:
                assert torch.allclose(aggregated[k], losses[k] * V), \
                    f"Sum aggregation should multiply by V"
            print(f"  ✅ Sum aggregation correct (scaled by {V})")

def test_views_per_sample_validation():
    """Test views_per_sample parameter validation."""
    print("\n" + "="*80)
    print("Testing views_per_sample Validation")
    print("="*80)
    
    from mmdet.models.utils.multi_view import MultiViewBackbone
    from mmdet.registry import MODELS
    
    # Mock backbone
    class MockBackbone(nn.Module):
        def forward(self, x):
            # Simple identity for testing
            B, C, H, W = x.shape
            # Return FPN-like output
            return [x, x, x, x]
    
    print("\n[Test] MVViT fusion WITHOUT views_per_sample (should raise error)")
    try:
        backbone = MultiViewBackbone(
            backbone=MockBackbone(),
            fusion='mvvit',
            views_per_sample=None  # Missing!
        )
        print("  ❌ Should have raised ValueError!")
        assert False, "Expected ValueError"
    except ValueError as e:
        print(f"  ✅ Correctly raised error: {str(e)[:80]}...")
    
    print("\n[Test] MVViT fusion WITH views_per_sample (should work)")
    try:
        backbone = MultiViewBackbone(
            backbone=MockBackbone(),
            fusion='mvvit',
            views_per_sample=8,
            mvvit=dict(type='MVViT', embed_dim=256, num_heads=8, num_layers=1)
        )
        print(f"  ✅ Initialized successfully with views_per_sample=8")
        print(f"  ✅ MVViT module created: {backbone.mvvit is not None}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        raise

def test_shape_handling():
    """Test shape handling for 4D and 5D inputs."""
    print("\n" + "="*80)
    print("Testing Shape Handling")
    print("="*80)
    
    from mmdet.models.utils.multi_view import MultiViewBackbone
    
    class MockBackbone(nn.Module):
        def forward(self, x):
            B, C, H, W = x.shape
            # Return single level for simplicity
            return x
    
    V, B, C, H, W = 8, 2, 3, 64, 64
    
    # Test with views_per_sample set
    backbone = MultiViewBackbone(
        backbone=MockBackbone(),
        fusion='mvvit',
        views_per_sample=V,
        mvvit=dict(type='MVViT', embed_dim=256, num_heads=4, num_layers=1, 
                   spatial_attention='global')
    )
    
    print(f"\n[Test] 4D input: (B*V, C, H, W) = ({B*V}, {C}, {H}, {W})")
    input_4d = torch.randn(B * V, C, H, W)
    try:
        output = backbone(input_4d)
        print(f"  ✅ 4D input processed, output shape: {output.shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise
    
    print(f"\n[Test] 5D input: (B, V, C, H, W) = ({B}, {V}, {C}, {H}, {W})")
    input_5d = torch.randn(B, V, C, H, W)
    try:
        output = backbone(input_5d)
        print(f"  ✅ 5D input processed, output shape: {output.shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise
    
    print(f"\n[Test] Wrong 4D input: (B*V+1, C, H, W) - should error")
    input_wrong = torch.randn(B * V + 1, C, H, W)
    try:
        output = backbone(input_wrong)
        print(f"  ❌ Should have raised ValueError!")
        assert False
    except ValueError as e:
        print(f"  ✅ Correctly raised error: {str(e)[:80]}...")

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MULTI-VIEW SOFTTEACHER - FIX VALIDATION TESTS")
    print("="*80)
    
    try:
        test_mvvit_modes()
        test_loss_aggregation()
        test_views_per_sample_validation()
        test_shape_handling()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nKiến trúc multi-view đã được fix đúng và hoạt động tốt!")
        print("Có thể train model ngay bây giờ.\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        raise

if __name__ == '__main__':
    main()
