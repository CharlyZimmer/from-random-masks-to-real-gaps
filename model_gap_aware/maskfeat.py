import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import nn

from typing import Optional, Tuple, cast
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from model_gap_aware.vivit import ViViT

class TemporalCrossAttentionDecoder(nn.Module):

    def __init__(self, embed_dim=128, patch_size=8, out_chans=1, num_frames=5, decoder_heads=1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=decoder_heads,
            batch_first=True,
        )
        self.decoder_pred = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size**2 * out_chans),
        )
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.num_frames = num_frames

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * channels)
        imgs: (N, channels, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, h * p))
        return imgs

    def forward(self, encoder_tokens):
        """
        encoder_tokens: (B, T, N, D)

        Returns:
            (B, C, H, W) where C == self.out_chans
        """
        B, T, N, D = encoder_tokens.shape
        mid = T // 2

        # Reshape to treat each spatial patch independently across time: (B*N, T, D)
        x = encoder_tokens.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Middle frame tokens as query; all frames as keys and values
        q = self.norm_q(x[:, mid:mid+1, :])   # (B*N, 1, D)
        kv = self.norm_kv(x)                   # (B*N, T, D)

        attn_out, _ = self.temporal_attn(q, kv, kv)   # (B*N, 1, D)
        pooled = attn_out.squeeze(1).reshape(B, N, D)  # (B, N, D)

        x = self.decoder_pred(pooled)   # (B, N, patch_size**2 * out_chans)
        x = self.unpatchify(x)          # (B, out_chans, H, W)

        return x



class MaskFeatGapFiller(pl.LightningModule):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_frames: int,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 192,
        depth: int = 1,
        heads: int = 1,
        decoder_heads: int = 1,
        mask_ratio: float = 0.7,
        lr: float = 1e-4,
        loss_type: str = "mae",
        warmup_steps: int = 500,
        total_steps: int = 10000,
        norm_range: Optional[Tuple[float, float]] = None,
        gap_only_masking: bool = False,
        structured_masking: bool = False,
        block_size_range: Tuple[float, float] = (0.67, 0.97),
        block_aspect_ratio_range: Tuple[float, float] = (0.3, 1/0.3),
    ):
        super().__init__()
        if loss_type not in ("mae", "mse"):
            raise ValueError(f"loss_type must be 'mae' or 'mse', got '{loss_type}'")
        self.save_hyperparameters()
        self.norm_range = norm_range
        self.patch_size = patch_size
        self.lr = lr
        self.loss_type = loss_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gap_only_masking = gap_only_masking
        self.encoder = ViViT(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_channels=in_channels,
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mask_ratio=mask_ratio,
            structured_masking=structured_masking,
            block_size_range=block_size_range,
            block_aspect_ratio_range=block_aspect_ratio_range,
        )
        self.decoder = TemporalCrossAttentionDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_chans=out_channels,
            num_frames=num_frames,
            decoder_heads=decoder_heads,
        )

    def forward(self, x, gap_only_masking: bool = False):
        enc_tokens, patch_mask = self.encoder(x, gap_only_masking=gap_only_masking)
        out = self.decoder(enc_tokens)
        return out, patch_mask

    def mae(self, preds, targets, patch_mask):
        """
        Masked Absolute Error
        preds: (B, C, H, W)
        targets: (B, C, H, W)
        patch_mask: (B, N) -- 1 means patch was removed
        valid_mask: (B, C, H, W) -- non-NaN values in target

        Notes:
            - patch_mask is defined per spatial patch, applied equally over all channels.
            - NaNs in targets are excluded from the loss.
        """

        if preds.dim() != 4 or targets.dim() != 4:
            raise ValueError(f"Expected preds/targets with shape (B,C,H,W). Got preds={tuple(preds.shape)}, targets={tuple(targets.shape)}")

        B, N = patch_mask.shape
        _, C, H, W = targets.shape

        ph = H // self.patch_size
        pw = W // self.patch_size

        # --- Ensure same dtype and device ---
        preds = preds.to(torch.float32)
        targets = targets.to(torch.float32)

        # --- Build mask: valid (non-NaN) pixels ---
        valid_mask = ~torch.isnan(targets)  # (B, C, H, W)
        targets = torch.nan_to_num(targets)

        # --- patch_mask to pixel level (B, H, W) ---
        patch_mask = patch_mask.reshape(B, ph, pw)
        patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=1)
        patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=2)

        # --- Broadcast spatial mask across channels -> (B, C, H, W) ---
        patch_mask = patch_mask.unsqueeze(1).expand(-1, C, -1, -1)

        # join both masks to obtain the area that is valid for loss computation
        loss_mask = patch_mask.bool() & valid_mask.bool()
        loss_mask = loss_mask.to(dtype=preds.dtype)

        loss = torch.abs(preds - targets)
        masked_loss = loss * loss_mask

        # --- Compute mean safely ---
        denom = loss_mask.sum().clamp(min=1)
        mae = masked_loss.sum() / denom

        return mae, loss_mask

    def mse(self, preds, targets, patch_mask):
        """
        Masked Squared Error — same masking logic as mae(), squared residual.
        """
        if preds.dim() != 4 or targets.dim() != 4:
            raise ValueError(f"Expected preds/targets with shape (B,C,H,W). Got preds={tuple(preds.shape)}, targets={tuple(targets.shape)}")

        B, N = patch_mask.shape
        _, C, H, W = targets.shape

        ph = H // self.patch_size
        pw = W // self.patch_size

        preds = preds.to(torch.float32)
        targets = targets.to(torch.float32)

        valid_mask = ~torch.isnan(targets)
        targets = torch.nan_to_num(targets)

        patch_mask = patch_mask.reshape(B, ph, pw)
        patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=1)
        patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=2)
        patch_mask = patch_mask.unsqueeze(1).expand(-1, C, -1, -1)

        loss_mask = patch_mask.bool() & valid_mask.bool()
        loss_mask = loss_mask.to(dtype=preds.dtype)

        loss = (preds - targets) ** 2
        masked_loss = loss * loss_mask

        denom = loss_mask.sum().clamp(min=1)
        mse = masked_loss.sum() / denom

        return mse, loss_mask

    def compute_loss(self, preds, targets, patch_mask):
        if self.loss_type == "mse":
            return self.mse(preds, targets, patch_mask)
        return self.mae(preds, targets, patch_mask)

    def rescaled_loss(self, preds, targets, patch_mask, norm_range):
        preds = preds * (norm_range[1] - norm_range[0]) + norm_range[0]
        targets = targets * (norm_range[1] - norm_range[0]) + norm_range[0]
        return self.compute_loss(preds, targets, patch_mask)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, patch_mask = self.forward(x, gap_only_masking=self.gap_only_masking)
        key = self.loss_type.upper()

        loss, _ = self.compute_loss(preds, y, patch_mask)
        self.log(key, loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.norm_range is not None:
            rescaled, _ = self.rescaled_loss(preds, y, patch_mask, self.norm_range)
            self.log(f"{key} Rescaled", rescaled, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, patch_mask = self.forward(x, gap_only_masking=self.gap_only_masking)
        key = self.loss_type.upper()

        loss, _ = self.compute_loss(preds, y, patch_mask)
        self.log(f"{key}_val", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.norm_range is not None:
            rescaled, _ = self.rescaled_loss(preds, y, patch_mask, self.norm_range)
            self.log(f"{key} Rescaled_val", rescaled, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, patch_mask = self.forward(x, gap_only_masking=self.gap_only_masking)
        key = self.loss_type.upper()

        loss, _ = self.compute_loss(preds, y, patch_mask)
        self.log(f"{key}_test", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.norm_range is not None:
            rescaled, _ = self.rescaled_loss(preds, y, patch_mask, self.norm_range)
            self.log(f"{key} Rescaled_test", rescaled, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)

        warmup_steps = int(self.warmup_steps)
        cosine_steps = max(1, int(self.total_steps) - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        cfg = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }
        return cast(OptimizerLRScheduler, cfg)



