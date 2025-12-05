import torch

from mmdet.datasets.wrappers.multi_view_dataset import MultiViewDataset, multi_view_collate


def test_multi_view_dataset_and_collate():
    # create simple groups: 2 samples, each with 3 views of 3x16x16
    g1 = [torch.randn(3, 16, 16) for _ in range(3)]
    g2 = [torch.randn(3, 16, 16) for _ in range(3)]
    ds = MultiViewDataset([g1, g2])
    item0 = ds[0]
    assert item0['inputs'].shape == (3, 3, 16, 16)
    collated = multi_view_collate([ds[0], ds[1]])
    assert collated['inputs'].shape == (2, 3, 3, 16, 16)