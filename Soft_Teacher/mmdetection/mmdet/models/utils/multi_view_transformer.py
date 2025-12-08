"""Multi-View Vision Transformer (MVViT) for refining features across views."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
from mmdet.registry import MODELS


@MODELS.register_module()
class MVViT(nn.Module):
    """Multi-View Vision Transformer for feature refinement across views.
    
    This module applies cross-view attention to refine features from multiple views.
    Each view's features interact with other views through multi-head attention.
    
    Architecture:
        For each feature level:
        - features_views: List[V] of (B, C, H, W)
        - Flatten spatial dims: (B, V, H*W, C)
        - Apply cross-view self-attention
        - Reshape back: List[V] of (B, C, H, W)
    
    Args:
        embed_dim (int): Feature embedding dimension (C). Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_layers (int): Number of transformer layers. Default: 2.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.1.
        use_layer_norm (bool): Use LayerNorm instead of BatchNorm. Default: True.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        spatial_attention: str = 'efficient',  # 'efficient', 'moderate', or 'global'
        use_gradient_checkpointing: bool = True  # Enable for 32GB GPU
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.spatial_attention = spatial_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Build transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_layers)
        ])
        
        # Optional positional encoding for views (learned)
        # We'll create this lazily when we know V
        self._view_pos_embed = None
        
        # Spatial positional embeddings (created lazily based on H, W)
        self._spatial_pos_embed = {}
        
        # Pooling and upsample queries (pre-allocated for common sizes)
        # Updated: Added K=2048 for better hard class detection (Severe_Rust, Chipped)
        self.pooling_queries = nn.ParameterDict({
            '256': nn.Parameter(torch.randn(256, embed_dim)),
            '512': nn.Parameter(torch.randn(512, embed_dim)),
            '1024': nn.Parameter(torch.randn(1024, embed_dim)),
            '2048': nn.Parameter(torch.randn(2048, embed_dim))  # NEW: For K=2048
        })
        # Initialize with normal distribution
        for param in self.pooling_queries.values():
            nn.init.normal_(param, std=0.02)
        
        # Upsample queries (pre-allocated for common FPN sizes)
        self.upsample_queries = nn.ParameterDict({
            '32x32': nn.Parameter(torch.randn(32*32, embed_dim)),    # P4 level
            '64x64': nn.Parameter(torch.randn(64*64, embed_dim)),    # P3 level  
            '128x128': nn.Parameter(torch.randn(128*128, embed_dim)), # P2 level
        })
        # Initialize with normal distribution
        for param in self.upsample_queries.values():
            nn.init.normal_(param, std=0.02)
        
        # Keep lazy dicts for fallback (rare sizes)
        self._pooling_queries = {}
        self._upsample_queries = {}
        
        # Projection layers for different input channels (created lazily)
        # FPN has different channels per level: 256, 512, 1024, 2048
        self._projections = nn.ModuleDict()
        
        # Learnable residual weight (instead of fixed 0.5)
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))
        
    def _get_projection(self, in_channels: int, device: torch.device):
        """Get or create projection layer for input channels -> embed_dim."""
        key = str(in_channels)
        if key not in self._projections:
            if in_channels == self.embed_dim:
                # No projection needed
                self._projections[key] = nn.Identity()
            else:
                # 1x1 conv to project channels
                proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                self._projections[key] = proj
        return self._projections[key].to(device)
        
    def _init_view_pos_embed(self, V: int, device: torch.device):
        """Lazily initialize view positional embeddings."""
        if self._view_pos_embed is None or self._view_pos_embed.shape[0] != V:
            # Create learnable positional embeddings for V views
            # Shape: (V, embed_dim)
            embed = nn.Parameter(torch.randn(V, self.embed_dim))
            nn.init.normal_(embed, std=0.02)
            self.register_parameter('view_pos_embed', embed)
            self._view_pos_embed = embed
        return self._view_pos_embed.to(device)
    
    def _get_spatial_pos_embed(self, H: int, W: int, device: torch.device):
        """Get or create spatial positional embeddings for given H, W."""
        key = f"{H}x{W}"
        if key not in self._spatial_pos_embed:
            # Create learned positional embeddings
            # Shape: (H*W, embed_dim)
            embed = nn.Parameter(torch.randn(H * W, self.embed_dim))
            nn.init.normal_(embed, std=0.02)
            self.register_parameter(f'spatial_pos_embed_{key}', embed)
            self._spatial_pos_embed[key] = embed
        return self._spatial_pos_embed[key].to(device)
    
    def _get_pooling_queries(self, K: int, device: torch.device):
        """Get pooling queries, preferring pre-allocated common sizes."""
        # Try pre-allocated sizes first
        if str(K) in self.pooling_queries:
            return self.pooling_queries[str(K)].to(device)
        
        # Fallback to lazy creation for uncommon sizes
        key = f"pool_{K}"
        if key not in self._pooling_queries:
            queries = nn.Parameter(torch.randn(K, self.embed_dim))
            nn.init.normal_(queries, std=0.02)
            self.register_parameter(f'pooling_queries_{K}', queries)
            self._pooling_queries[key] = queries
        return self._pooling_queries[key].to(device)
    
    def _get_upsample_queries(self, H: int, W: int, device: torch.device):
        """Get upsample queries, preferring pre-allocated common sizes."""
        key = f"{H}x{W}"
        
        # Try pre-allocated sizes first
        if key in self.upsample_queries:
            return self.upsample_queries[key].to(device)
        
        # Fallback to lazy creation for uncommon sizes
        if key not in self._upsample_queries:
            queries = nn.Parameter(torch.randn(H * W, self.embed_dim))
            nn.init.normal_(queries, std=0.02)
            self.register_parameter(f'upsample_queries_{key}', queries)
            self._upsample_queries[key] = queries
        return self._upsample_queries[key].to(device)
    
    def forward_single_level(
        self, 
        features_views: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply MVViT on features from a single FPN level.
        
        Supports two attention modes (optimized for 32GB GPU):
        1. 'global': Pool spatial dims, attention on V tokens only 
           - Memory: ~1-2GB, Speed: Fast, Accuracy: Moderate
           - Use when: Very limited memory or quick experiments
        
        2. 'efficient': Downsample spatial to ~256 tokens/view, then attention on V*256 tokens
           - Memory: ~4-8GB, Speed: Medium, Accuracy: Good (RECOMMENDED)
           - Use when: Normal training with 32GB GPU
        
        NOTE: 'full' mode removed - it causes OOM on 32GB GPU with typical image sizes!
        
        Works with any input channels by using adaptive projection.
        
        Args:
            features_views: List of V tensors, each (B, C, H, W)
            
        Returns:
            List of V tensors, each (B, C, H, W) - refined features
        """
        V = len(features_views)
        if V == 0:
            return []
        
        # Get shape info from first view
        B, C, H, W = features_views[0].shape
        device = features_views[0].device
        
        # DEBUG: Log MVViT execution
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        
        # if self._forward_count <= 3 or self._forward_count % 100 == 0:
        #     from mmengine.logging import print_log
        #     print_log(
        #         f"[MVViT] Forward #{self._forward_count}: V={V}, B={B}, C={C}, "
        #         f"H={H}x{W}, mode={self.spatial_attention}", 
        #         logger='current'
        #     )
        
        # Stack views: (B, V, C, H, W)
        stacked = torch.stack(features_views, dim=1)
        
        # Project to embed_dim if needed
        proj = self._get_projection(C, device)
        if C != self.embed_dim:
            # Reshape for conv: (B*V, C, H, W)
            stacked_flat = stacked.view(B * V, C, H, W)
            projected = proj(stacked_flat).view(B, V, self.embed_dim, H, W)
        else:
            projected = stacked
        
        # Apply different attention strategies
        if self.spatial_attention == 'global':
            # Global pooling: (B, V, embed_dim)
            view_tokens = projected.mean(dim=[3, 4])
            view_pos = self._init_view_pos_embed(V, device)
            view_tokens = view_tokens + view_pos.unsqueeze(0)
            
            # Cross-view attention with gradient checkpointing
            tokens = view_tokens
            if self.use_gradient_checkpointing and self.training:
                for layer in self.layers:
                    tokens = torch.utils.checkpoint.checkpoint(
                        layer, tokens, use_reentrant=False
                    )
            else:
                for layer in self.layers:
                    tokens = layer(tokens)
            
            # Broadcast back to spatial
            refined = tokens.unsqueeze(-1).unsqueeze(-1).expand(B, V, self.embed_dim, H, W)
            
        elif self.spatial_attention == 'efficient':
            # Efficient mode: Downsample spatial for attention
            # Target: ~256 spatial tokens per view
            target_tokens = 256
            scale = min(1.0, (target_tokens / (H * W)) ** 0.5)
            
            if scale < 1.0:
                # Downsample
                new_h = max(1, int(H * scale))
                new_w = max(1, int(W * scale))
                # (B*V, C, H, W) -> (B*V, C, new_h, new_w)
                downsampled = F.interpolate(
                    projected.view(B * V, self.embed_dim, H, W),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).view(B, V, self.embed_dim, new_h, new_w)
            else:
                downsampled = projected
                new_h, new_w = H, W
            
            # Flatten spatial: (B, V*new_h*new_w, embed_dim)
            tokens = downsampled.permute(0, 1, 3, 4, 2).reshape(B, V * new_h * new_w, self.embed_dim)
            
            # Add positional embeddings
            view_pos = self._init_view_pos_embed(V, device)  # (V, embed_dim)
            spatial_pos = self._get_spatial_pos_embed(new_h, new_w, device)  # (new_h*new_w, embed_dim)
            
            # Combine: view_pos[v] + spatial_pos[hw] for each position
            pos_embed = (
                view_pos.unsqueeze(1).expand(V, new_h * new_w, self.embed_dim) +
                spatial_pos.unsqueeze(0).expand(V, new_h * new_w, self.embed_dim)
            ).reshape(V * new_h * new_w, self.embed_dim)
            tokens = tokens + pos_embed.unsqueeze(0)
            
            # Cross-view spatial attention with gradient checkpointing
            if self.use_gradient_checkpointing and self.training:
                for layer in self.layers:
                    tokens = torch.utils.checkpoint.checkpoint(
                        layer, tokens, use_reentrant=False
                    )
            else:
                for layer in self.layers:
                    tokens = layer(tokens)
            
            # Reshape back: (B, V, new_h, new_w, embed_dim)
            refined_downsampled = tokens.view(B, V, new_h, new_w, self.embed_dim).permute(0, 1, 4, 2, 3)
            
            # Upsample back to original size if needed
            if scale < 1.0:
                refined = F.interpolate(
                    refined_downsampled.view(B * V, self.embed_dim, new_h, new_w),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).view(B, V, self.embed_dim, H, W)
            else:
                refined = refined_downsampled
        
        elif self.spatial_attention == 'moderate':
            # Moderate mode: Attention Pooling BEFORE transformer
            # Strategy: Pool H*W -> K tokens per view, then cross-view attention on V*K tokens
            # This ACTUALLY reduces FLOPs unlike pooling after transformer!
            # Memory: ~8-16GB, Speed: Medium, Accuracy: Good (RECOMMENDED for 32GB GPU)
            
            # ADAPTIVE K: Balance memory vs information loss
            # Strategy: Keep 15-25% spatial detail, capped for memory safety
            # 
            # Analysis of information loss:
            # FPN Level | H×W     | Tokens | K=512 | K=1024 | K=1536 (NEW)
            # --------- | ------- | ------ | ----- | ------ | ------------
            # P2        | 180×64  | 11,520 | 4.4%  | 8.9%   | 13.3% ✅
            # P3        | 90×32   | 2,880  | 17.8% | 35.6%  | 53.3% ✅
            # P4        | 45×16   | 720    | 71.1% | 100%   | 100%  ✅
            # P5        | 23×8    | 184    | 100%  | 100%   | 100%  ✅
            #
            # K=1536 gives:
            # - P2: 13.3% detail (good for narrow defects like Severe_Rust, Chipped)
            # - P3/P4/P5: No downsampling (preserves ALL spatial info!)
            # - Memory: 8-16GB (safe for 32GB GPU)
            # - Tradeoff: Slightly less detail on P2 than K=2048 (13.3% vs 17.8%)
            #   but MUCH more stable and won't OOM!
            K = min(1536, H * W)
            
            # Step 1: Flatten spatial for each view independently
            # (B, V, embed_dim, H, W) -> (B*V, H*W, embed_dim)
            tokens = projected.permute(0, 1, 3, 4, 2).reshape(B * V, H * W, self.embed_dim)
            
            # Step 2: Add spatial positional embeddings
            spatial_pos = self._get_spatial_pos_embed(H, W, device)  # (H*W, embed_dim)
            tokens = tokens + spatial_pos.unsqueeze(0)  # (B*V, H*W, embed_dim)
            
            # Step 3: Attention Pooling - Pool each view H*W -> K tokens
            # Get K learnable pooling queries (cached per K)
            pooling_queries = self._get_pooling_queries(K, device)  # (K, embed_dim)
            pooling_queries = pooling_queries.unsqueeze(0).expand(B * V, K, self.embed_dim)  # (B*V, K, embed_dim)
            
            # Compute attention: Q=pooling_queries (K queries), K=V=tokens (H*W keys)
            # (B*V, K, embed_dim) x (B*V, embed_dim, H*W) -> (B*V, K, H*W)
            attn_scores = torch.matmul(pooling_queries, tokens.transpose(1, 2))
            attn_scores = attn_scores / (self.embed_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B*V, K, H*W)
            
            # Apply attention: (B*V, K, H*W) x (B*V, H*W, embed_dim) -> (B*V, K, embed_dim)
            pooled_tokens = torch.matmul(attn_weights, tokens)  # (B*V, K, embed_dim)
            
            # Step 4: Reshape to separate views: (B*V, K, embed_dim) -> (B, V*K, embed_dim)
            pooled_tokens = pooled_tokens.reshape(B, V * K, self.embed_dim)
            
            # Step 5: Add view positional embeddings
            view_pos = self._init_view_pos_embed(V, device)  # (V, embed_dim)
            view_pos_expanded = view_pos.unsqueeze(1).expand(V, K, self.embed_dim).reshape(V * K, self.embed_dim)
            pooled_tokens = pooled_tokens + view_pos_expanded.unsqueeze(0)
            
            # Step 6: Cross-view attention on V*K tokens (MUCH smaller than V*H*W!)
            # This is where we save FLOPs: O((V*K)^2) << O((V*H*W)^2)
            if self.use_gradient_checkpointing and self.training:
                for layer in self.layers:
                    pooled_tokens = torch.utils.checkpoint.checkpoint(
                        layer, pooled_tokens, use_reentrant=False
                    )
            else:
                for layer in self.layers:
                    pooled_tokens = layer(pooled_tokens)
            
            # Step 7: Reshape back: (B, V*K, embed_dim) -> (B, V, K, embed_dim)
            refined_pooled = pooled_tokens.reshape(B, V, K, self.embed_dim)
            
            # Step 8: Broadcast K tokens back to H*W spatial locations
            # Use attention to upsample (learnable)
            # For each spatial location (h,w), attend to K pooled tokens
            upsample_queries = self._get_upsample_queries(H, W, device)  # (H*W, embed_dim)
            upsample_queries = upsample_queries.unsqueeze(0).unsqueeze(0).expand(B, V, H * W, self.embed_dim)  # (B, V, H*W, embed_dim)
            
            # Reshape for attention: (B*V, H*W, embed_dim)
            upsample_queries = upsample_queries.reshape(B * V, H * W, self.embed_dim)
            refined_pooled_flat = refined_pooled.reshape(B * V, K, self.embed_dim)
            
            # Attention: Q=upsample_queries (H*W queries), K=V=refined_pooled (K keys)
            # (B*V, H*W, embed_dim) x (B*V, embed_dim, K) -> (B*V, H*W, K)
            up_attn_scores = torch.matmul(upsample_queries, refined_pooled_flat.transpose(1, 2))
            up_attn_scores = up_attn_scores / (self.embed_dim ** 0.5)
            up_attn_weights = F.softmax(up_attn_scores, dim=-1)  # (B*V, H*W, K)
            
            # Apply attention: (B*V, H*W, K) x (B*V, K, embed_dim) -> (B*V, H*W, embed_dim)
            upsampled = torch.matmul(up_attn_weights, refined_pooled_flat)
            
            # Reshape back: (B*V, H*W, embed_dim) -> (B, V, embed_dim, H, W)
            refined = upsampled.reshape(B, V, H, W, self.embed_dim).permute(0, 1, 4, 2, 3)
            
            # Step 9: Residual connection with original features
            refined = refined + projected
        
        else:
            raise ValueError(
                f"spatial_attention='{self.spatial_attention}' is not supported. "
                f"Use 'global', 'efficient', or 'moderate'. "
                f"Note: 'full' mode removed due to OOM on 32GB GPU!"
            )
        
        # Project back to original channels if needed
        if C != self.embed_dim:
            back_key = f'back_proj_{C}'
            if back_key not in self._projections:
                back_proj = nn.Conv2d(self.embed_dim, C, kernel_size=1, bias=False)
                nn.init.xavier_uniform_(back_proj.weight)
                self._projections[back_key] = back_proj
            back_proj = self._projections[back_key].to(device)
            
            refined = back_proj(
                refined.view(B * V, self.embed_dim, H, W)
            ).view(B, V, C, H, W)
        
        # Residual connection with learnable weight
        output = self.residual_alpha * refined + (1 - self.residual_alpha) * stacked
        
        # Split back into list of views
        refined_views = [output[:, v] for v in range(V)]
        
        return refined_views
    
    def forward(
        self, 
        features_views: Union[List[torch.Tensor], List[List[torch.Tensor]]]
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass for MVViT.
        
        Args:
            features_views: Either:
                - List of V tensors (single level): each (B, C, H, W)
                - List of V lists of L tensors (multi-level FPN): 
                  features_views[v][l] is (B, C_l, H_l, W_l)
        
        Returns:
            Refined features in the same structure as input
        """
        if len(features_views) == 0:
            return features_views
        
        # Check if multi-level (FPN) or single level
        if isinstance(features_views[0], (list, tuple)):
            # Multi-level case: features_views is List[V] of List[L]
            # We need to process each level independently
            V = len(features_views)
            L = len(features_views[0])
            
            # Reorganize to List[L] of List[V]
            level_views = []
            for l in range(L):
                level_views.append([features_views[v][l] for v in range(V)])
            
            # Apply MVViT to each level
            refined_levels = []
            for level_feats in level_views:
                refined_levels.append(self.forward_single_level(level_feats))
            
            # Reorganize back to List[V] of List[L]
            refined_views = []
            for v in range(V):
                refined_views.append([refined_levels[l][v] for l in range(L)])
            
            return refined_views
        else:
            # Single level case: features_views is List[V] of (B, C, H, W)
            return self.forward_single_level(features_views)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention and FFN.
    
    Args:
        d_model (int): Feature dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
        use_layer_norm (bool): Use LayerNorm (True) or BatchNorm (False).
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.use_layer_norm = use_layer_norm
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            src: (B, N, C) where N is number of tokens
            
        Returns:
            (B, N, C) - refined features
        """
        # Self-attention with residual
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        
        if self.use_layer_norm:
            src = self.norm1(src)
        else:
            # BatchNorm expects (B, C, N)
            src = src.transpose(1, 2)
            src = self.norm1(src)
            src = src.transpose(1, 2)
        
        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        if self.use_layer_norm:
            src = self.norm2(src)
        else:
            src = src.transpose(1, 2)
            src = self.norm2(src)
            src = src.transpose(1, 2)
        
        return src
