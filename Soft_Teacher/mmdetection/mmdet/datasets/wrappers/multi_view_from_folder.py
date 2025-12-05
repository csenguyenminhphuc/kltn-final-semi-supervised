import os
from typing import Callable, List, Optional, Tuple
import re
import psutil
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import json
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
import torch
import torch.nn.functional as F
from mmdet.registry import DATASETS
from mmengine.dataset import Compose



def default_filename_parser(filename: str) -> Tuple[str, int]:
    stem = filename.split('.')[0]
    idx1 = re.search(r"(S\d+)", stem)
    idx2 =  re.search(r"(bright|dark)_\d", stem)
    idx_crop = re.search(r"crop_(\d)", stem)
    if idx1 is None or idx2 is None or idx_crop is None:
        raise ValueError(f'Filename {filename} does not match expected pattern')
    image_id = idx1.group(0)+ '_' + idx2.group(0)
    idx = int(idx_crop.group(0).split('_')[1])

    return image_id, idx


@DATASETS.register_module()
class MultiViewFromFolder(Dataset):
    """Dataset that groups crop images in a folder into multi-view samples.

    Args:
        root (str): folder containing crop image files.
        filename_parser (callable): parse filename -> (image_id, crop_idx).
        views_per_sample (int, optional): desired number of views per sample.
            If a sample has fewer views, it will be padded with zeros. If it
            has more, it will be truncated to this number. If None, keep all.
        transform (callable, optional): applied to each image Tensor.
    """

    def __init__(self,
                 root: Optional[str] = None,
                 filename_parser: Callable[[str], Tuple[str, int]] = default_filename_parser,
                 views_per_sample: Optional[int] = 9,
                 transform: Optional[Callable] = None,
                 ann_file: Optional[str] = None,
                 pipeline: Optional[Callable] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Optional[dict] = None,
                 metainfo: Optional[dict] = None,
                 test_mode: bool = False,
                 backend_args: Optional[dict] = None,
                 **kwargs):
        # support mmdet/mmengine dataset build signature by accepting common
        # dataset kwargs (pipeline, data_root, data_prefix, metainfo, ...)
        # Determine root: prefer explicit root, else combine data_root and data_prefix
        if root is None:
            if data_root is not None:
                if isinstance(data_prefix, dict) and data_prefix.get('img'):
                    root = os.path.join(data_root, data_prefix.get('img'))
                else:
                    root = data_root
        self.root = root
        self.filename_parser = filename_parser
        self.views_per_sample = views_per_sample
        # Lazy-init Compose pipeline configuration
        self.pipeline_cfg = pipeline
        self.transform = None
        self._is_compose = isinstance(pipeline, list) and Compose is not None

        # store metainfo if provided (use _metainfo to avoid clashing with property)
        self._metainfo = metainfo or {}
        # dataset initialization flag for compatibility with mmengine BaseDataset
        self._fully_initialized = False
        # optional annotations mapping (filename -> list of ann dicts)
        self.ann_map = None
        # path to annotation json if provided
        self.anno_file = None
        # category_id to label index mapping (COCO format uses 1-based IDs)
        self.cat2label = {}

        # Initialize empty groups - will be populated from COCO JSON
        self.groups = []
        self.group_ids = []
        self.group_shapes = []

        # Load annotations and build groups from COCO JSON
        if ann_file is not None:
            self.load_coco_annotations(ann_file)
            # Build groups directly from COCO JSON (no filename_parser needed)
            self._build_groups_from_coco()
        else:
            raise ValueError("ann_file is required - dataset must be built from COCO JSON with base_img_id")

    def __len__(self):
        return len(self.groups)
 
    def get_data_info(self, idx: int) -> dict:
        """Return a lightweight data info dict for compatibility.

        This mirrors the minimal keys used by samplers/filters: 'img_path',
        'img_id', 'width', 'height', and 'instances' (list).
        """
        if idx < 0 or idx >= len(self.groups):
            raise IndexError('index out of range')
        paths = self.groups[idx]
        # prefer first non-None path as representative
        img_path = ''
        for p in paths:
            if p is not None:
                img_path = p
                break
        img_id = self.group_ids[idx] if len(self.group_ids) > idx else str(idx)
        if hasattr(self, 'group_shapes') and len(self.group_shapes) > idx:
            w, h = self.group_shapes[idx]
        else:
            w, h = 0, 0
        instances = []
        # if ann_map available, collect anns from the representative file
        if self.ann_map is not None and img_path:
            fname = os.path.basename(img_path)
            anns = self.ann_map.get(fname, [])
            for a in anns:
                x, y, bw, bh = a.get('bbox', [0, 0, 0, 0])
                instances.append({'bbox': [x, y, x + bw, y + bh],
                                  'bbox_label': a.get('category_id', 0)})

        return {
            'img_path': img_path,
            'img_id': img_id,
            'width': w,
            'height': h,
            'instances': instances,
        }

    @property
    def metainfo(self) -> dict:
        """Return meta information expected by mmengine datasets."""
        # return a shallow copy to avoid external mutation
        return dict(self._metainfo) if self._metainfo is not None else {}

    def full_init(self) -> None:
        """Compatibility full_init called by dataset wrappers.

        We already performed necessary scanning in __init__, so just mark
        as initialized. If an annotation file path was provided but not
        loaded yet, load it now.
        """
        if self._fully_initialized:
            return
        if getattr(self, 'anno_file', None) and self.ann_map is None:
            try:
                self.load_coco_annotations(self.anno_file)
            except Exception:
                # ignore load failures here; caller may handle missing anns
                pass
        self._fully_initialized = True

    def _load_image(self, path: Optional[str]) -> torch.Tensor:
        import cv2  # Move cv2 import here to avoid copying OpenCV state
        if path is None:
            return torch.zeros(3, 1, 1, dtype=torch.uint8)

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            return torch.zeros(3, 1, 1, dtype=torch.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)
        return t

    # def _load_image(self, path: Optional[str]) -> torch.Tensor:
    #     print("Loading image:", path, flush=True)
    #     if path is None:
    #         return torch.zeros(3, 1, 1, dtype=torch.float)
    #     t = read_image(path).float()  # returns CxHxW uint8 -> float
    #     return t

    def load_coco_annotations(self, coco_json_path: str):
        """Load a COCO-style json and create a filename -> annotations map.

        The COCO `file_name` field should match the crop filenames in the
        dataset (basename). Each annotation dict is kept as-is.
        """
        if not os.path.exists(coco_json_path):
            raise FileNotFoundError(coco_json_path)
        with open(coco_json_path, 'r') as f:
            j = json.load(f)
        # build image id -> file_name and reverse mapping
        id2name = {im['id']: im['file_name'] for im in j.get('images', [])}
        self.name2id = {im['file_name']: im['id'] for im in j.get('images', [])}
        
        # Build filename -> base_img_id mapping from COCO JSON
        self.name2base_img_id = {}
        self.name2crop_num = {}
        for im in j.get('images', []):
            fname = im['file_name']
            base_img_id = im.get('base_img_id')
            crop_num = im.get('crop_num')
            if base_img_id is not None:
                self.name2base_img_id[fname] = base_img_id
            if crop_num is not None:
                self.name2crop_num[fname] = crop_num
        
        # Build cat2label mapping: COCO category_id -> label index (0-based)
        # Get category IDs and sort them to create consistent mapping
        categories = j.get('categories', [])
        cat_ids = sorted([cat['id'] for cat in categories])
        # Filter out category_id=0 if it exists and is unused (like 'drill' supercategory)
        # Then enumerate to create 0-based index: {1:0, 2:1, 3:2, 4:3, 5:4}
        filtered_cat_ids = [cat_id for cat_id in cat_ids if cat_id > 0]
        if filtered_cat_ids:
            self.cat2label = {cat_id: i for i, cat_id in enumerate(filtered_cat_ids)}
        else:
            # Fallback: use all category IDs including 0
            self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        
        ann_map = {}
        for ann in j.get('annotations', []):
            img_id = ann['image_id']
            fname = id2name.get(img_id)
            if fname is None:
                continue
            ann_map.setdefault(fname, []).append(ann)
        self.ann_map = ann_map
        self.anno_file = coco_json_path
        
        # DEBUG: Log annotation loading stats
        from mmengine.logging import print_log
        print_log(f"[COCO ANNO] Loaded from: {coco_json_path}", logger='current')
        print_log(f"  → {len(j.get('images', []))} images, {len(j.get('annotations', []))} annotations", logger='current')
        print_log(f"  → {len(ann_map)} files have annotations", logger='current')
        print_log(f"  → {len(self.name2base_img_id)} files have base_img_id", logger='current')
        print_log(f"  → Categories: {cat_ids}", logger='current')
        print_log(f"  → cat2label mapping: {self.cat2label}", logger='current')
        # Show first 3 filenames with base_img_id
        for i, (fname, anns) in enumerate(list(ann_map.items())[:3]):
            base_id = self.name2base_img_id.get(fname, 'N/A')
            crop_num = self.name2crop_num.get(fname, 'N/A')
            print_log(f"  → File '{fname}': {len(anns)} boxes, base_img_id={base_id}, crop_num={crop_num}", logger='current')

    def _build_groups_from_coco(self):
        """Build groups based on base_img_id from COCO JSON.
        
        This groups all crops with the same base_img_id together,
        sorted by crop_num (1-8).
        """
        if not hasattr(self, 'name2base_img_id') or not self.name2base_img_id:
            raise ValueError("COCO JSON must have base_img_id field for all images")
        
        from mmengine.logging import print_log
        from collections import defaultdict
        
        # Build mapping: base_img_id -> [(crop_num, filepath), ...]
        base_img_groups = defaultdict(list)
        
        # Iterate through all files in COCO JSON
        for fname, base_id in self.name2base_img_id.items():
            crop_num = self.name2crop_num.get(fname, 0)
            filepath = os.path.join(self.root, fname)
            
            # Only add if file exists
            if os.path.exists(filepath):
                base_img_groups[base_id].append((crop_num, filepath))
            else:
                print_log(f"[WARNING] File not found: {filepath}", logger='current')
        
        # Build groups: sort by crop_num and pad/truncate to views_per_sample
        self.groups = []
        self.group_ids = []
        
        for base_id in sorted(base_img_groups.keys()):
            crops = base_img_groups[base_id]
            # Sort by crop_num
            crops_sorted = sorted(crops, key=lambda x: x[0])
            paths = [p for (_, p) in crops_sorted]
            
            # Pad or truncate to views_per_sample
            if self.views_per_sample is not None:
                if len(paths) < self.views_per_sample:
                    paths = paths + [None] * (self.views_per_sample - len(paths))
                elif len(paths) > self.views_per_sample:
                    paths = paths[:self.views_per_sample]
            
            self.groups.append(paths)
            self.group_ids.append(base_id)
        
        self.group_shapes = [(0, 0)] * len(self.groups)
        
        print_log(f"[COCO] Built {len(self.groups)} groups from base_img_id", logger='current')
        print_log(f"[COCO] Total files in COCO JSON: {len(self.name2base_img_id)}", logger='current')
        print_log(f"[COCO] Unique base_img_ids: {len(set(self.name2base_img_id.values()))}", logger='current')
        
        # Show first 3 groups
        for i in range(min(3, len(self.groups))):
            base_id = self.group_ids[i]
            paths = self.groups[i]
            num_views = sum(1 for p in paths if p is not None)
            print_log(f"  → Group {i}: base_img_id={base_id}, {num_views}/{len(paths)} views", logger='current')

    def __getitem__(self, idx):
        paths = self.groups[idx]
        group_id = self.group_ids[idx] if hasattr(self, 'group_ids') and len(self.group_ids) > idx else str(idx)
        views = []
        data_samples = []
        for view_idx, p in enumerate(paths):
            # If we're using a Compose pipeline (config-style pipeline built
            # into a callable), avoid preloading the image into memory here.
            # Let the pipeline's LoadImageFromFile run inside the worker and
            # read the file on demand. For non-Compose callables, load the
            # image tensor and apply the transform if provided.
            t = None
            orig_h, orig_w = 0, 0

            # Lazy-init Compose pipeline
            if self._is_compose and self.transform is None:
                from mmengine.dataset import Compose
                self.transform = Compose(self.pipeline_cfg)

            if getattr(self, '_is_compose', False):
                # Try to get basic shape without reading the full image into
                # a large numpy array. Use PIL.Image.open which is light-weight
                # for getting size and closes the file immediately.
                if p is not None:
                    try:
                        with Image.open(p) as im:
                            w, h = im.size
                            orig_h, orig_w = int(h), int(w)
                    except Exception:
                        orig_h, orig_w = 0, 0
                
                # Get COCO img_id from filename BEFORE pipeline
                coco_img_id_early = None
                if p is not None:
                    fname_early = os.path.basename(p)
                    if hasattr(self, 'name2id'):
                        coco_img_id_early = self.name2id.get(fname_early)
                
                # Use numeric COCO img_id if available, else string fallback
                img_id_for_pipeline = coco_img_id_early if coco_img_id_early is not None else f"{group_id}_view{view_idx}"

                results = {
                    'img_path': p,
                    'ori_shape': (orig_h, orig_w, 3),
                    'img_id': img_id_for_pipeline,  # Use numeric COCO ID
                }

                results = self.transform(results)

                # Find a tensor in the transform output (common keys or nested).
                def _find_tensor(x):
                    if isinstance(x, torch.Tensor):
                        return x
                    if isinstance(x, np.ndarray):
                        return torch.from_numpy(x)
                    if isinstance(x, dict):
                        for k in ('img', 'inputs', 'image'):
                            if k in x:
                                v = x[k]
                                if isinstance(v, (torch.Tensor, np.ndarray)):
                                    return v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                        for v in x.values():
                            found = _find_tensor(v)
                            if found is not None:
                                return found
                        return None
                    if isinstance(x, (list, tuple)):
                        for v in x:
                            found = _find_tensor(v)
                            if found is not None:
                                return found
                        return None
                    return None

                found = _find_tensor(results)
                if isinstance(found, torch.Tensor):
                    t = found
                elif isinstance(found, np.ndarray):
                    t = torch.from_numpy(found)
                else:
                    inp = results.get('inputs') if isinstance(results, dict) else None
                    if isinstance(inp, (torch.Tensor, np.ndarray)):
                        t = inp if isinstance(inp, torch.Tensor) else torch.from_numpy(inp)
                    else:
                        # No tensor found in results; defer to a light-weight
                        # load now (will be small since it's one image) so we
                        # always return a tensor for the view.
                        t = self._load_image(p)
                        if p is not None:
                            orig_h, orig_w = int(t.shape[1]), int(t.shape[2])

            else:
                # Non-compose transform path: load image tensor and apply
                # the provided callable transform (if any). This keeps
                # behaviour compatible with simple torchvision transforms.
                t = self._load_image(p)
                if p is not None:
                    orig_h, orig_w = int(t.shape[1]), int(t.shape[2])
                
                # Get COCO img_id from filename BEFORE transform
                coco_img_id_early = None
                if p is not None:
                    fname_early = os.path.basename(p)
                    if hasattr(self, 'name2id'):
                        coco_img_id_early = self.name2id.get(fname_early)
                
                # Use numeric COCO img_id if available, else string fallback
                img_id_for_transform = coco_img_id_early if coco_img_id_early is not None else f"{group_id}_view{view_idx}"
                
                if self.transform is not None:
                    try:
                        # Some transforms expect a dict, some expect a tensor/PIL.
                        # Try tensor first; if it fails, try passing a dict.
                        t = self.transform(t)
                    except Exception:
                        try:
                            res = {'img': t, 'ori_shape': (orig_h, orig_w, 3), 'img_id': img_id_for_transform}
                            out = self.transform(res)
                            found = None
                            if isinstance(out, dict):
                                found = out.get('img') or out.get('inputs')
                            if isinstance(found, (torch.Tensor, np.ndarray)):
                                t = found if isinstance(found, torch.Tensor) else torch.from_numpy(found)
                        except Exception:
                            # if transform fails, keep original tensor
                            pass

            # Ensure we always have a 3-dim tensor for the view and in C,H,W
            if t is None:
                t = torch.zeros(3, 1, 1, dtype=torch.uint8)

            # Normalize tensor shape to (C, H, W). Some pipelines return
            # HWC numpy arrays or tensors — convert to CHW.
            if isinstance(t, torch.Tensor):
                if t.ndim == 3:
                    # common cases: (C,H,W) or (H,W,C)
                    if t.shape[0] in (1, 3):
                        pass  # already (C,H,W)
                    elif t.shape[2] in (1, 3):
                        # (H,W,C) -> permute
                        t = t.permute(2, 0, 1).contiguous()
                    else:
                        # Unknown layout, try to coerce by moving channels
                        t = t.permute(2, 0, 1).contiguous()
                elif t.ndim == 2:
                    # single-channel H,W -> add channel dim
                    t = t.unsqueeze(0)
                else:
                    # fallback: reshape to (C, H, W) with C=3 if possible
                    try:
                        t = t.view(3, t.shape[-2], t.shape[-1])
                    except Exception:
                        t = torch.zeros(3, 1, 1, dtype=torch.uint8)

            views.append(t)
            # explicitly delete temporary references to help GC
            try:
                del found
            except Exception:
                pass
            del t

            # prepare per-view metainfo
            if p is not None:
                fname = os.path.basename(p)
            else:
                fname = None

            # Get COCO image_id from filename if available
            coco_img_id = None
            if fname and hasattr(self, 'name2id'):
                coco_img_id = self.name2id.get(fname)
            
            # Extract base_img_id (without crop suffix) for grouping
            # Try to get from COCO JSON first, fallback to group_id from filename_parser
            base_img_id = group_id  # Default to group_id from filename_parser
            if fname and hasattr(self, 'name2base_img_id'):
                coco_base_img_id = self.name2base_img_id.get(fname)
                if coco_base_img_id is not None:
                    base_img_id = coco_base_img_id
            
            # Get crop_num from COCO JSON if available
            crop_num = None
            if fname and hasattr(self, 'name2crop_num'):
                crop_num = self.name2crop_num.get(fname)
            
            # For COCO evaluation:
            # - img_id: COCO image_id for this specific crop (for matching GT)
            # - base_img_id: Group ID to aggregate predictions from all crops
            if coco_img_id is not None:
                # Use COCO image_id for this crop
                img_id = coco_img_id  # Keep as int for COCO compatibility
            else:
                # Training fallback: use string ID with view suffix
                img_id = f"{group_id}_view{view_idx}"

            # build DetDataSample per view; ensure metainfo always present
            # ensure gt_instances attribute always exists (may be empty)
            ds = DetDataSample()
            inst0 = InstanceData()
            # initialize empty tensors so downstream code can safely access
            inst0.bboxes = torch.zeros((0, 4), dtype=torch.float32)
            inst0.labels = torch.zeros((0, ), dtype=torch.long)
            ds.gt_instances = inst0
            ds_meta = dict(
                file_name=fname,
                img_id=img_id,  # COCO image_id for this specific crop
                base_img_id=base_img_id,  # Group ID for aggregating multi-view predictions
                view_id=view_idx,
                crop_num=crop_num,  # Crop number from COCO JSON (1-8)
                ori_shape=(orig_h, orig_w, 3),
                # include img_shape to satisfy downstream heads (H, W, C)
                img_shape=(orig_h, orig_w, 3),
                # Add default scale_factor for validation (will be updated by Resize transform)
                scale_factor=(1.0, 1.0),
            )

            if self.ann_map is not None and p is not None:
                anns = self.ann_map.get(fname, [])
                if len(anns) > 0:
                    # convert COCO bbox [x,y,w,h] -> [x1,y1,x2,y2]
                    # and map category_id to 0-based label index
                    bboxes = []
                    labels = []
                    cat2label = getattr(self, 'cat2label', {})
                    for a in anns:
                        x, y, w, h = a['bbox']
                        bboxes.append([x, y, x + w, y + h])
                        # Map COCO category_id (e.g., 1-5) to label index (0-4)
                        cat_id = a.get('category_id', 0)
                        label_idx = cat2label.get(cat_id, cat_id)
                        labels.append(label_idx)
                    
                    # DEBUG: Log first 3 samples to verify bbox loading
                    if not hasattr(self, '_bbox_debug_count'):
                        self._bbox_debug_count = 0
                    if self._bbox_debug_count < 3:
                        from mmengine.logging import print_log
                        self._bbox_debug_count += 1
                        print_log(f"[BBOX DEBUG {self._bbox_debug_count}] File: {fname}, Image shape: {orig_h}x{orig_w}", logger='current')
                        print_log(f"  → Loaded {len(bboxes)} boxes from COCO JSON", logger='current')
                        for i, (bbox, lbl) in enumerate(zip(bboxes[:3], labels[:3])):
                            print_log(f"    Box {i}: {bbox}, label: {lbl}", logger='current')
                        if len(anns) == 0:
                            print_log(f"  → WARNING: No annotations in ann_map for '{fname}'", logger='current')
                    
                    inst0.bboxes = torch.tensor(bboxes, dtype=torch.float32)
                    inst0.labels = torch.tensor(labels, dtype=torch.long)
                    ds.gt_instances = inst0

            # attach homography matrix (identity) by default so downstream
            # algorithms (e.g. SoftTeacher) can safely call
            # torch.from_numpy(ds.homography_matrix).
            ds.homography_matrix = np.eye(3, dtype=np.float32)
            ds.set_metainfo(ds_meta)
            data_samples.append(ds)
        # Memory-efficient: pad views one at a time
        # First pass: find max dimensions
        max_h = max(v.shape[1] for v in views)
        max_w = max(v.shape[2] for v in views)
        
        # Second pass: pad and collect
        views_padded = []
        for v in views:
            c, h, w = v.shape
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h == 0 and pad_w == 0:
                views_padded.append(v)
            else:
                padded = F.pad(v, (0, pad_w, 0, pad_h), 'constant', 0)
                views_padded.append(padded)
        
        stacked = torch.stack(views_padded, dim=0)
        del views, views_padded
        return {'inputs': stacked, 'data_samples': data_samples}

def multi_view_collate_flatten(batch):
    # batch: list of dicts, each dict: {'inputs': (V,C,H,W), 'data_samples': maybe list}
    # Separate labeled (sup) and unlabeled (unsup) items
    sup_items = []
    unsup_items = []
    for item in batch:
        labeled = False
        for ds in item.get('data_samples', []):
            gt = getattr(ds, 'gt_instances', None)
            # consider labeled only if gt is InstanceData and contains bboxes
            if isinstance(gt, InstanceData):
                bboxes = getattr(gt, 'bboxes', None)
                try:
                    if bboxes is not None and int(bboxes.numel()) > 0:
                        labeled = True
                        break
                except Exception:
                    pass
        if labeled:
            sup_items.append(item)
        else:
            unsup_items.append(item)

    def _pack_nested(items):
        if len(items) == 0:
            return None, []
        
        # Memory-efficient: process one item at a time
        # First pass: find max dimensions
        max_h = 0
        max_w = 0
        for it in items:
            x = it['inputs']
            max_h = max(max_h, x.shape[2])
            max_w = max(max_w, x.shape[3])
        
        # Second pass: pad and collect
        padded_list = []
        nested_ds = []
        
        for it in items:
            x = it['inputs']  # (V, C, H, W)
            V, C, H, W = x.shape
            pad_h = max_h - H
            pad_w = max_w - W
            
            # Pad if necessary
            if pad_h > 0 or pad_w > 0:
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
            else:
                x_padded = x
            padded_list.append(x_padded)
            
            # Handle data_samples
            ds = it.get('data_samples')
            if ds is None or ds == {}:
                # create placeholder DetDataSample with empty InstanceData
                placeholders = []
                for _ in range(V):
                    placeholder = DetDataSample()
                    inst0 = InstanceData()
                    inst0.bboxes = torch.zeros((0, 4), dtype=torch.float32)
                    inst0.labels = torch.zeros((0, ), dtype=torch.long)
                    placeholder.gt_instances = inst0
                    # ensure homography exists on placeholders too
                    placeholder.homography_matrix = np.eye(3, dtype=np.float32)
                    placeholders.append(placeholder)
                nested_ds.append(placeholders)
            elif isinstance(ds, list):
                nested_ds.append(ds)
            else:
                nested_ds.append([ds for _ in range(V)])
        
        # Stack only when needed
        batched = torch.stack(padded_list, dim=0)  # (B, V, C, H, W)
        return batched, nested_ds

    sup_batched, sup_nested = _pack_nested(sup_items)
    unsup_batched, unsup_nested = _pack_nested(unsup_items)

    # convert batched (B,V,C,H,W) to list of independent per-view tensors
    def _flatten_batched(batched):
        if batched is None:
            return []
        B, V, C, H, W = batched.shape
        # Reshape and create independent tensors (not slices)
        flat = batched.reshape(B * V, C, H, W)
        # Use unbind or split to create independent tensors
        return list(torch.unbind(flat, dim=0))

    sup_list = _flatten_batched(sup_batched) if sup_batched is not None else []
    unsup_list = _flatten_batched(unsup_batched) if unsup_batched is not None else []

    # flatten nested ds lists (per-sample lists of per-view DetDataSample)
    def _flatten_nested_ds(nested):
        if not nested:
            return []
        flat = []
        for per in nested:
            flat.extend(per)
        return flat

    sup_ds_flat = _flatten_nested_ds(sup_nested) if sup_nested else []
    unsup_ds_flat = _flatten_nested_ds(unsup_nested) if unsup_nested else []

    # Only include branches that contain at least one item. If a branch has
    # zero items, omit it entirely so downstream DetDataPreprocessor does not
    # receive an empty tensor list (which would trigger stack_batch assertion).
    inputs_out = {}
    data_sample_out = {}

    if len(sup_list) > 0:
        inputs_out['sup'] = sup_list
        data_sample_out['sup'] = sup_ds_flat

    if len(unsup_list) > 0:
        # For unsupervised branches, both teacher and student expect the
        # same flattened unsup list.
        inputs_out['unsup_teacher'] = unsup_list
        inputs_out['unsup_student'] = unsup_list
        data_sample_out['unsup_teacher'] = unsup_ds_flat
        data_sample_out['unsup_student'] = unsup_ds_flat

    # Use plural 'data_samples' to match mmengine/mmdet expectations.
    return {'inputs': inputs_out, 'data_samples': data_sample_out}


def multi_view_collate(batch):
    """Backward-compatible alias expected by some imports.

    This implementation delegates to `multi_view_collate_flatten` to
    provide a consistent multi-branch output. If a nested (B, V, C, H, W)
    representation is required, consider replacing this implementation
    with a more specific collate that returns per-sample grouped tensors.
    """
    return multi_view_collate_flatten(batch)


def multi_view_collate_val(batch):
    """Collate function for validation/test that flattens multi-view samples.
    
    IMPORTANT: Flatten to (B*V, C, H, W) for data_preprocessor compatibility.
    MultiViewBackbone will reshape back to (B, V, ...) for cross-view attention.
    
    Args:
        batch: list of dicts, each {'inputs': (V,C,H,W), 'data_samples': list of V DetDataSample}
    
    Returns:
        dict with:
            'inputs': List of (C,H,W) tensors, length B*V
            'data_samples': Flattened list of DetDataSample, length B*V
    """
    if len(batch) == 0:
        return {'inputs': [], 'data_samples': []}
    
    # Memory-efficient validation collate: process one item at a time
    if not batch:
        return {'inputs': [], 'data_samples': []}
    
    # First pass: find max dimensions
    max_h = 0
    max_w = 0
    for item in batch:
        x = item['inputs']
        max_h = max(max_h, x.shape[2])
        max_w = max(max_w, x.shape[3])
    
    # Second pass: pad and flatten directly to final lists
    inputs_flat = []
    data_samples_flat = []
    
    for item in batch:
        x = item['inputs']  # (V, C, H, W)
        V, C, H, W = x.shape
        pad_h = max_h - H
        pad_w = max_w - W
        
        # Pad if necessary
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
        else:
            x_padded = x
        
        # Unbind views and add to flat list
        views_unbound = torch.unbind(x_padded, dim=0)
        inputs_flat.extend(views_unbound)
        
        # Flatten data_samples
        ds_list = item.get('data_samples', [])
        if isinstance(ds_list, list):
            data_samples_flat.extend(ds_list)
        else:
            data_samples_flat.append(ds_list)
    
    return {'inputs': inputs_flat, 'data_samples': data_samples_flat}


