import torch
import torch.nn as nn
from typing import Optional, List
from mmdet.registry import MODELS
from mmengine.logging import print_log


@MODELS.register_module()
class MultiViewBackbone(nn.Module):
    """Wrapper that applies a backbone on flattened views and fuses features.

    Supports fusion strategies via the ``fusion`` argument. Input images have
    shape (B, V, C, H, W). The backbone is applied on flattened
    (B*V, C, H, W). The returned features from backbone can be a dict,
    list/tuple or a single tensor; this wrapper will reshape each feature map
    back to (B, V, C_f, H_f, W_f) and fuse over the view dimension.

    Args:
        backbone (nn.Module): the shared backbone to apply per-view.
        fusion (str): one of {'mean', 'max', 'weighted', 'concat', 'mvvit', 'none'}. Default 'mean'.
        concat_reduce (bool): when fusion='concat', if True create a 1x1 conv
            to reduce concatenated channels back to C_f. The conv is created
            lazily on the first forward when channel sizes are known.
        mvvit (dict, optional): Config for MVViT module when fusion='mvvit'.
    """

    def __init__(self, backbone: nn.Module, fusion: str = 'mean', concat_reduce: bool = False, 
                 mvvit: Optional[dict] = None, views_per_sample: Optional[int] = None):
        super().__init__()
        # Accept either an already-built nn.Module or a config (dict/ConfigDict)
        # describing the backbone. If a config is provided, build the module
        # via the registry so `self.backbone` is always a callable nn.Module.
        if not isinstance(backbone, nn.Module):
            # Accept either a direct backbone cfg or a containing dict like
            # {'backbone': {...}, 'fusion': ..., ...} which is commonly
            # provided when building from config files. In the latter case,
            # extract the inner backbone cfg.
            cfg = backbone
            if isinstance(cfg, dict) and 'backbone' in cfg:
                cfg = cfg['backbone']
            try:
                backbone = MODELS.build(cfg)
            except Exception:
                # re-raise with additional context
                raise
        self.backbone = backbone
        assert fusion in {'mean', 'max', 'weighted', 'concat', 'mvvit', 'none'}, f'Unsupported fusion: {fusion}'
        self.fusion = fusion
        self.concat_reduce = concat_reduce
        
        # Store views_per_sample for reshaping flattened inputs
        self._views_per_sample = views_per_sample
        
        # Validate views_per_sample when MVViT is used
        if fusion == 'mvvit' and views_per_sample is None:
            raise ValueError(
                "MultiViewBackbone with fusion='mvvit' REQUIRES "
                "views_per_sample to be set in config. Please add "
                "views_per_sample parameter to backbone config."
            )

        # Build MVViT module if fusion='mvvit'
        self.mvvit = None
        if fusion == 'mvvit':
            if mvvit is None:
                # Use default MVViT config
                mvvit = dict(type='MVViT', embed_dim=256, num_heads=8, num_layers=2)
            self.mvvit = MODELS.build(mvvit)
            print_log(
                f"MultiViewBackbone: Initialized with MVViT fusion, "
                f"views_per_sample={views_per_sample}", 
                logger='current'
            )

        # lazy params created at first forward when V / channel sizes are known
        # do not use variable annotations here to avoid style/compat issues
        self._weights = None  # for 'weighted' fusion: nn.Parameter of shape (V,)
        self._reduce_convs = None  # for 'concat' fusion: will be nn.ModuleDict

    def _init_weights_param(self, V: int):
        # create learnable per-view weights initialized uniform
        if self._weights is None:
            w = torch.ones(V, dtype=torch.float)
            w = w / float(V)
            param = nn.Parameter(w)
            # register under a stable name and keep a reference
            self.register_parameter('view_weights', param)
            self._weights = param

    def forward(self, x: torch.Tensor):
        """Forward pass for MultiViewBackbone.
        
        Args:
            x: Input tensor. Can be either:
                - (B, V, C, H, W): Explicit multi-view format
                - (B*V, C, H, W): Flattened format (from data_preprocessor)
                
        Returns:
            Features in same format as standard backbone (for compatibility)
        """
        # DEBUG: Log backbone execution
        if not hasattr(self, '_backbone_forward_count'):
            self._backbone_forward_count = 0
        self._backbone_forward_count += 1
        
        # if self._backbone_forward_count <= 3 or self._backbone_forward_count % 100 == 0:
        #     print_log(f"[MultiViewBackbone] Forward #{self._backbone_forward_count}: input shape={x.shape}, fusion={self.fusion}, V={getattr(self, '_views_per_sample', 'unset')}", logger='current')
        
        # Case 1: Explicit multi-view format (B, V, C, H, W)
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            flat = x.view(B * V, C, H, W)
            feats = self.backbone(flat)
            
            # if self._backbone_forward_count <= 3:
            #     print_log(f"  → 5D input: B={B}, V={V}, flattened to ({B*V},{C},{H},{W})", logger='current')
            
        # Case 2: Flattened format (B*V, C, H, W) - need to infer B and V
        elif x.dim() == 4:
            # For flattened input, we need to know views_per_sample to reshape
            # This should be set from config or inferred
            BV, C, H, W = x.shape
            
            # If mvvit fusion, we MUST reshape to multi-view format
            if self.fusion == 'mvvit':
                # Get V from stored config
                V = self._views_per_sample
                if V is None:
                    raise ValueError(
                        "MultiViewBackbone with fusion='mvvit' received 4D input "
                        f"with shape {x.shape} but views_per_sample is not set. "
                        "Please ensure views_per_sample is specified in backbone config."
                    )
                
                # Ensure BV is divisible by V
                if BV % V != 0:
                    raise ValueError(
                        f"Input batch size {BV} is not divisible by views_per_sample {V}. "
                        f"Expected (B*{V}, C, H, W) format."
                    )
                    
                B = BV // V
                flat = x
                feats = self.backbone(flat)
                
                # if self._backbone_forward_count <= 3 or self._backbone_forward_count % 100 == 0:
                #     print_log(f"  → 4D input with mvvit: BV={BV}, V={V}, B={B}", logger='current')
            else:
                # For non-mvvit fusion, treat as standard batch
                flat = x
                B, V = BV, 1  # Treat as single view
                feats = self.backbone(flat)
                
                # if self._backbone_forward_count <= 3 or self._backbone_forward_count % 100 == 0:
                #     print_log(f"  → 4D input without mvvit: treating as standard batch", logger='current')
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # For MVViT fusion, we need to preserve per-view features
        if self.fusion == 'mvvit':
            # if self._backbone_forward_count <= 3 or self._backbone_forward_count % 100 == 0:
            #     print_log(f"  Applying MVViT fusion...", logger='current')
            result = self._apply_mvvit_fusion(feats, B, V)
            # if self._backbone_forward_count <= 3 or self._backbone_forward_count % 100 == 0:
            #     if isinstance(result, (list, tuple)):
            #         print_log(f"  MVViT output: {len(result)} levels, first level shape: {result[0].shape}", logger='current')
            #     else:
            #         print_log(f"  MVViT output shape: {result.shape}", logger='current')
            return result

        def _fuse_fmap(fmap: torch.Tensor, level_idx: Optional[int] = None, level_key: Optional[str] = None):
            # fmap: (B*V, C_f, H_f, W_f)
            bf, Cf, Hf, Wf = fmap.shape
            fmap = fmap.view(B, V, Cf, Hf, Wf)
            if self.fusion == 'mean':
                return fmap.mean(dim=1)
            elif self.fusion == 'max':
                return fmap.max(dim=1).values
            elif self.fusion == 'none':
                # do not fuse: return flattened per-view features (B*V, C_f, H_f, W_f)
                return fmap.view(B * V, Cf, Hf, Wf)
            elif self.fusion == 'weighted':
                # lazy init weights
                if self._weights is None:
                    self._init_weights_param(V)
                weights = torch.softmax(self._weights, dim=0)  # (V,)
                w = weights.view(1, V, 1, 1, 1).to(fmap.device)
                return (fmap * w).sum(dim=1)
            elif self.fusion == 'concat':
                # concat channels: (B, V*Cf, Hf, Wf)
                cat = fmap.view(B, V * Cf, Hf, Wf)
                if self.concat_reduce:
                    # lazily create a reduce conv for this level if missing
                    if self._reduce_convs is None:
                        # use ModuleDict keyed by level to support dict outputs
                        self._reduce_convs = nn.ModuleDict()
                    key = level_key if level_key is not None else str(level_idx)
                    if key not in self._reduce_convs:
                        conv = nn.Conv2d(V * Cf, Cf, kernel_size=1)
                        self._reduce_convs[key] = conv
                    conv = self._reduce_convs[key]
                    return conv(cat)
                else:
                    return cat
            else:
                raise RuntimeError(f'Unknown fusion: {self.fusion}')

        # support dict, list/tuple, or single tensor
        if isinstance(feats, dict):
            fused = {}
            for k, fmap in feats.items():
                fused[k] = _fuse_fmap(fmap, level_key=k)
            return fused
        elif isinstance(feats, (list, tuple)):
            fused_list = []
            for i, fmap in enumerate(feats):
                fused_list.append(_fuse_fmap(fmap, level_idx=i))
            return fused_list
        else:
            # single tensor
            return _fuse_fmap(feats)
    
    def _apply_mvvit_fusion(self, feats, B: int, V: int):
        """Apply MVViT to refine multi-view features.
        
        Args:
            feats: Features from backbone, can be dict/list/tensor
            B: Batch size
            V: Number of views
            
        Returns:
            Refined features keeping per-view structure
        """
        # Organize features as List[V] of features per view
        if isinstance(feats, dict):
            # Dict of feature levels: {level_key: (B*V, C, H, W)}
            # Reorganize to List[V] of {level_key: (B, C, H, W)}
            level_keys = list(feats.keys())
            views_feats = []
            for v in range(V):
                view_dict = {}
                for key in level_keys:
                    fmap = feats[key]  # (B*V, C, H, W)
                    _, C, H, W = fmap.shape
                    fmap_reshaped = fmap.view(B, V, C, H, W)
                    view_dict[key] = fmap_reshaped[:, v]  # (B, C, H, W)
                views_feats.append(view_dict)
            
            # For dict features, we need to apply MVViT per level
            # Convert to List[V] of List[L] format
            views_lists = []
            for v in range(V):
                views_lists.append([views_feats[v][key] for key in level_keys])
            
            # Apply MVViT: List[V] of List[L] -> List[V] of List[L]
            refined_views = self.mvvit(views_lists)
            
            # Convert back to dict format: List[V] of {level_key: (B, C, H, W)}
            refined_dict_views = []
            for v in range(V):
                view_dict = {level_keys[l]: refined_views[v][l] for l in range(len(level_keys))}
                refined_dict_views.append(view_dict)
            
            # Flatten back for standard processing: {level_key: (B*V, C, H, W)}
            result = {}
            for key in level_keys:
                level_feats = [refined_dict_views[v][key] for v in range(V)]  # List[V] of (B, C, H, W)
                stacked = torch.stack(level_feats, dim=1)  # (B, V, C, H, W)
                result[key] = stacked.reshape(B * V, *stacked.shape[2:])  # (B*V, C, H, W)
            return result
            
        elif isinstance(feats, (list, tuple)):
            # List of feature levels: [(B*V, C_l, H_l, W_l)]
            L = len(feats)
            views_feats = []
            for v in range(V):
                view_list = []
                for l in range(L):
                    fmap = feats[l]  # (B*V, C, H, W)
                    _, C, H, W = fmap.shape
                    fmap_reshaped = fmap.view(B, V, C, H, W)
                    view_list.append(fmap_reshaped[:, v])  # (B, C, H, W)
                views_feats.append(view_list)
            
            # Apply MVViT: List[V] of List[L] -> List[V] of List[L]
            refined_views = self.mvvit(views_feats)
            
            # Flatten back: List[L] of (B*V, C, H, W)
            result = []
            for l in range(L):
                level_feats = [refined_views[v][l] for v in range(V)]  # List[V] of (B, C, H, W)
                stacked = torch.stack(level_feats, dim=1)  # (B, V, C, H, W)
                result.append(stacked.reshape(B * V, *stacked.shape[2:]))  # (B*V, C, H, W)
            return result
            
        else:
            # Single tensor: (B*V, C, H, W)
            _, C, H, W = feats.shape
            feats_reshaped = feats.view(B, V, C, H, W)
            views_feats = [feats_reshaped[:, v] for v in range(V)]  # List[V] of (B, C, H, W)
            
            # Apply MVViT: List[V] -> List[V]
            refined_views = self.mvvit(views_feats)
            
            # Flatten back: (B*V, C, H, W)
            stacked = torch.stack(refined_views, dim=1)  # (B, V, C, H, W)
            return stacked.reshape(B * V, C, H, W)