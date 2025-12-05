from typing import List, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """A simple multi-view dataset wrapper.

    Args:
        groups (List[List[Union[str, np.ndarray, torch.Tensor]]]): A list where
            each element is a list of views. Each view may be a file path (str),
            a numpy array, or a torch.Tensor with shape (C,H,W).
        transform (callable, optional): A transform applied to each view. It
            should accept a torch.Tensor or numpy array and return a torch.Tensor
            with shape (C,H,W).
    """

    def __init__(self, groups: List[List[Union[str, np.ndarray, torch.Tensor]]], transform: Callable = None):
        self.groups = groups
        self.transform = transform

    def __len__(self):
        return len(self.groups)

    def _to_tensor(self, img) -> torch.Tensor:
        # img can be str path, numpy array HWC or CHW, or torch.Tensor
        if isinstance(img, str):
            # avoid heavy io dependency in this simple wrapper; user can pass arrays
            from PIL import Image

            arr = np.array(Image.open(img))
        elif isinstance(img, np.ndarray):
            arr = img
        elif torch.is_tensor(img):
            t = img
            if t.ndim == 3:
                return t
            elif t.ndim == 2:
                return t.unsqueeze(0)
            else:
                return t
        else:
            raise TypeError('Unsupported image type: {}'.format(type(img)))

        # convert HWC -> CHW
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        group = self.groups[idx]
        views = []
        for v in group:
            t = self._to_tensor(v)
            if self.transform is not None:
                t = self.transform(t)
            views.append(t)
        # stack to (V, C, H, W)
        views = [v if v.ndim == 3 else v.unsqueeze(0) for v in views]
        stacked = torch.stack(views, dim=0)
        return {'inputs': stacked, 'data_samples': None}


# def multi_view_collate(batch: List[dict]) -> dict:
#     """Collate function to stack multi-view samples into (B, V, C, H, W)."""
#     inputs = [item['inputs'] for item in batch]
#     # each inputs is (V, C, H, W)
#     batched = torch.stack(inputs, dim=0)
#     data_samples = [item.get('data_samples') for item in batch]
#     return {'inputs': batched, 'data_samples': data_samples}
def multi_view_collate(batch):
    # batch: list of dicts, each dict: {'inputs': (V,C,H,W), 'data_samples': maybe dict or list}
    inputs = [item['inputs'] for item in batch]             # list of (V,C,H,W)
    batched_inputs = torch.stack(inputs, dim=0)             # (B,V,C,H,W)

    # build nested data_samples: if item['data_samples'] is dict -> duplicate for V or keep per-view list if provided
    nested = []
    for item in batch:
        ds = item.get('data_samples')
        if ds is None or ds == {}:
            # no annotations: create list of empty dicts for each view
            nested.append([{} for _ in range(batched_inputs.shape[1])])
        elif isinstance(ds, list):
            # assume ds is already per-view list of length V
            nested.append(ds)
        else:
            # single dict for whole sample -> duplicate per view
            nested.append([ds for _ in range(batched_inputs.shape[1])])

    return {'inputs': batched_inputs, 'data_samples': nested}