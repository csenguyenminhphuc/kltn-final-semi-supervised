# Lightweight package initializer for mmdet.datasets.wrappers
# Expose MultiView dataset wrappers so imports like
# `from mmdet.datasets.wrappers.multi_view_from_folder import ...` work.
from .multi_view_from_folder import MultiViewFromFolder, multi_view_collate, multi_view_collate_flatten
from .multi_view_dataset import MultiViewDataset

__all__ = ['MultiViewFromFolder', 'multi_view_collate', 'multi_view_collate_flatten', 'MultiViewDataset']
