import sys, os, time
from torch.utils.data import DataLoader
import torch
sys.path.insert(0, 'mmdetection')
sys.path.insert(0, 'mmengine')

from mmdet.datasets.wrappers.multi_view_from_folder import MultiViewFromFolder, multi_view_collate

root = '/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/data_drill/train'

# count files in folder
files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root,f))]
num_files = len(files)
print('num files in folder:', num_files)

# build dataset
ds = MultiViewFromFolder(root, views_per_sample=9)
ds.load_coco_annotations('/home/coder/trong/KLTN_SEMI/code/Soft_Teacher/data_drill/anno_train/_annotations.coco.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(
    ds,
    batch_size=9,             # tune theo GPU mem
    shuffle=False,
    num_workers=2,            # tune theo CPU cores; 4-16 thường hợp lý
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=multi_view_collate
)

# Simple prefetcher to move next batch to GPU asynchronously
class Prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.next_batch = None

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is not None:
            # move inputs async
            with torch.cuda.stream(self.stream):
                self.next_batch['inputs'] = self.next_batch['inputs'].to(self.device, non_blocking=True)
                # if you have tensors in data_samples, move them similarly

    def __iter__(self):
        self.preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

# Usage
prefetcher = Prefetcher(loader, device)
cnt = 0
for batch in prefetcher:
    inputs = batch['inputs']  # already on GPU
    print(inputs.shape)  # should be (B, V, C, H, W)
    # perform on-GPU ops, e.g. normalization + model forward
    # inputs shape: (B, V, C, H, W)
    data_samples = batch['data_samples']  # list of lists of dicts
    # data_samples is a list of length B, each item is a list of length V
    # each inner item is a dict (empty if no annotations)
    # e.g. data_samples[0][0] is the annotations dict for view
    # of the first sample in the batch
    sample = data_samples[0][0]

    # In meta info
    print("=== META INFO ===")
    print(sample.metainfo)

    # In các data field hiện có
    print("\n=== DATA FIELDS ===")
    print(sample.keys())
    cnt += 1
    if cnt > 2:
        break
print('done')

