import math
import random

import numpy as np
import torch
from torch import nn
from einops import rearrange

from model_gap_aware.module import Attention, PreNorm, FeedForward

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 192, depth = 4, heads = 4,  in_channels = 2,  mask_ratio=0.7,
                 norm_layer=nn.LayerNorm, emb_dropout = 0., dim_head = 64, dropout = 0., scale_dim = 4,
                 structured_masking: bool = False, block_size_range = (0.67, 0.97), block_aspect_ratio_range = (0.3, 1/0.3),):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.dim = dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))

        self.norm = norm_layer(dim)

        self.mask_ratio = mask_ratio

        self.structured_masking = structured_masking
        self.block_size_range = block_size_range
        self.block_aspect_ratio_range = block_aspect_ratio_range

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))

        self.to_patch = nn.Linear(patch_dim, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

    def to_patch_embedding(self, x, patch_size):

        # (B, T, C, H, W) -> (B, T, N, patch_dim)
        # patch_dim = p1 * p2 * C
        x = rearrange(x, 'b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size)

        # create mask indicating patches that contain gaps: (B, T, N)
        patches_with_gaps = torch.isnan(x).any(dim=-1)

        x = torch.nan_to_num(x, nan=0.0)
        x = self.to_patch(x)

        return x, patches_with_gaps

    def random_masking(self, x, mask_ratio, forced_mask, gap_only_masking: bool = False):
        """
        Fully vectorized 3D MAE-style masking with forced masking and
        random per-sample mask ratio ∈ [0.7, 1.0].

        x: [N, T, L, D]
        forced_mask: [N, T, L]  (1 = forced masked)
        gap_only_masking: if True, skip random masking — only gap patches are masked.
        """
        assert x.dim() == 4
        device = x.device
        N, T, L, D = x.shape
        assert D == self.dim
        forced_mask = forced_mask.to(device)
        mid = T // 2

        # ---------------------------------------------------------------------------
        # 1) Per-sample mask ratios
        # ---------------------------------------------------------------------------
        if gap_only_masking:
            mask_ratio_sample = torch.zeros(N, device=device)
        else:
            mask_ratio_sample = torch.empty(N, device=device).uniform_(mask_ratio, 1.0)  # (N,)
        # t = torch.full(N, self.mask_ratio, device=device)
        len_keep_mid = (L * (1.0 - mask_ratio_sample)).long().clamp(min=0)     # (N,)

        # ---------------------------------------------------------------------------
        # 2) Forced mask for the middle frame
        # ---------------------------------------------------------------------------
        forced_mid = forced_mask[:, mid, :].bool()  # (N, L)

        # ---------------------------------------------------------------------------
        # 3) Generate noise and push forced positions to +inf so they sort last
        # ---------------------------------------------------------------------------
        noise = torch.rand(N, L, device=device)
        noise = noise.masked_fill(forced_mid, float("inf"))

        # Shuffle indexes per sample
        ids_shuffle_mid = torch.argsort(noise, dim=1)          # (N, L)
        ids_restore_mid = torch.argsort(ids_shuffle_mid, dim=1)  # (N, L)

        # ---------------------------------------------------------------------------
        # 4) Vectorized construction of sorted mask for all samples
        # ---------------------------------------------------------------------------
        # mid_mask_sorted: shape (N, L), before unshuffle
        # For each sample i, positions < len_keep_mid[i] are 0, others 1.
        # Create a row vector [0..L-1]
        positions = torch.arange(L, device=device)[None, :]  # (1, L)

        # Broadcasted comparison: keep=0, mask=1
        mid_mask_sorted = (positions >= len_keep_mid[:, None]).long()  # (N, L)

        # ---------------------------------------------------------------------------
        # 5) Unshuffle mask
        # ---------------------------------------------------------------------------
        mid_mask = torch.gather(mid_mask_sorted, 1, ids_restore_mid)  # (N, L)

        # Preserve forced masked positions
        mid_mask = mid_mask | forced_mid

        # ---------------------------------------------------------------------------
        # 6) Insert into full mask
        # ---------------------------------------------------------------------------
        full_mask = forced_mask.clone().long()  # (N, T, L)
        full_mask[:, mid, :] = mid_mask

        return full_mask

    def block_masking(self, x, forced_mask, gap_only_masking: bool = False):
        """
        BEiT-style (Bao et al.) single-block masking of the middle frame.

        Per sample: a target block area (as a fraction of the L-patch grid)
        is drawn from `self.block_size_range`. On a square grid, a rectangle
        can only reach area ratio `s` while holding aspect ratio `a` if
        `min(a, 1/a) >= s` (the smaller side is capped by the grid edge) — so
        an aspect ratio far from 1 geometrically cannot reach a high area
        target. The aspect ratio is therefore drawn log-uniformly from the
        sub-range of `self.block_aspect_ratio_range` that is actually
        feasible for the sampled size (`[size_ratio, 1/size_ratio]` clipped
        to the configured bounds), so the realized block reliably covers
        close to `size_ratio` — at the cost of aspect ratios collapsing
        toward 1 (square) as `size_ratio` approaches 1. Both dimensions are
        then scaled down proportionally (not clamped independently) if
        rounding pushes either past the grid, which preserves the sampled
        aspect ratio rather than silently distorting it.

        Patches already masked by the gap mask contribute no loss signal, so
        only patches newly covered by the block ("novel" patches) are useful.
        If a placement covers fewer than 10% of the grid in novel patches,
        it is resampled (up to 10 tries total, BEiT's own convention); the
        attempt with the highest novel count is kept if none clears the bar.

        x: [N, T, L, D]
        forced_mask: [N, T, L]  (1 = forced masked, i.e. gap)
        gap_only_masking: if True, skip block masking — only gap patches are masked.
        """
        assert x.dim() == 4
        device = x.device
        N, T, L, D = x.shape
        assert D == self.dim
        forced_mask = forced_mask.to(device)
        mid = T // 2

        if gap_only_masking:
            return forced_mask.clone().long()

        grid = int(round(math.sqrt(L)))
        assert grid * grid == L, "block masking requires a square patch grid"

        forced_mid_np = forced_mask[:, mid, :].detach().cpu().numpy().astype(bool)  # (N, L)
        min_novel = max(1, int(round(0.1 * L)))
        max_tries = 10
        aspect_lo, aspect_hi = self.block_aspect_ratio_range

        mid_mask_np = np.zeros((N, L), dtype=bool)
        for i in range(N):
            gap_i = forced_mid_np[i].reshape(grid, grid)
            best_block = None
            best_novel = -1
            for _ in range(max_tries):
                size_ratio = random.uniform(*self.block_size_range)
                target_area = size_ratio * L

                # Restrict aspect ratio to the sub-range that can actually
                # reach `size_ratio` on a square grid (collapses to a single
                # point, aspect == 1, as size_ratio approaches 1).
                feasible_lo = max(aspect_lo, size_ratio)
                feasible_hi = min(aspect_hi, 1.0 / size_ratio)
                if feasible_hi <= feasible_lo:
                    aspect = feasible_lo
                else:
                    aspect = math.exp(random.uniform(math.log(feasible_lo), math.log(feasible_hi)))

                h_ideal = math.sqrt(target_area * aspect)
                w_ideal = math.sqrt(target_area / aspect)
                # Scale both dimensions down together (preserving the sampled
                # aspect ratio) if either would otherwise exceed the grid.
                scale = min(1.0, grid / h_ideal, grid / w_ideal)
                h = min(max(int(round(h_ideal * scale)), 1), grid)
                w = min(max(int(round(w_ideal * scale)), 1), grid)
                top = random.randint(0, grid - h)
                left = random.randint(0, grid - w)

                block = np.zeros((grid, grid), dtype=bool)
                block[top:top + h, left:left + w] = True
                novel = int(np.count_nonzero(block & ~gap_i))

                if novel > best_novel:
                    best_novel = novel
                    best_block = block
                if novel >= min_novel:
                    break

            mid_mask_np[i] = (best_block | gap_i).reshape(L)

        mid_mask = torch.from_numpy(mid_mask_np).to(device=device, dtype=torch.long)
        full_mask = forced_mask.clone().long()  # (N, T, L)
        full_mask[:, mid, :] = mid_mask

        return full_mask

    def forward(self, x, gap_only_masking: bool = False):

        x, forced_mask = self.to_patch_embedding(x, self.patch_size)
        b, t, n, _ = x.shape

        if self.structured_masking:
            full_mask = self.block_masking(x, forced_mask, gap_only_masking=gap_only_masking)
        else:
            full_mask = self.random_masking(x, self.mask_ratio, forced_mask, gap_only_masking=gap_only_masking)

        mask_expanded = full_mask.unsqueeze(-1).bool()  # (B, T, N, 1)

        # Broadcast mask_token to (B, T, N, D)
        x = torch.where(mask_expanded, self.mask_token, x)

        x += self.pos_embedding
        x = self.dropout(x)

        # factorised self-attention
        # acts across time for each spatial location
        # achieved by reshaping (ViViT paper section 3.3. Model 3)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x, '(b t) ... -> b t ...', b=b)

        x = rearrange(x, 'b t n d -> (b n) t d', b=b)

        x = self.temporal_transformer(x)

        x = rearrange(x, '(b n) t d -> b t n d', b=b)

        x = self.norm(x)

        mid = t // 2
        mid_mask = full_mask[:, mid, :]

        return x, mid_mask






