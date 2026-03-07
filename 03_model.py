"""
Phase 2 — Model Architecture
==============================
File: pipeline/03_model.py

Architecture:
  Siamese ResNet-34 encoder (weight-shared)
      → bi-temporal feature difference at each scale
      → decoder with HAND-gated attention at every skip connection
      → flood probability output (B, 1, H, W)

MC Dropout is embedded in the decoder and stays ACTIVE at inference.
Never call model.eval() during uncertainty estimation — use enable_dropout().

Ablation variants controlled by config flags:
  use_hand_gate=False, hand_as_band=False  →  Variant A: SAR only baseline
  use_hand_gate=False, hand_as_band=True   →  Variant B: HAND as extra band
  use_hand_gate=True,  hand_as_band=False  →  Variant C: HAND gate, no UQ
  use_hand_gate=True  + MC dropout T>1     →  Variant D: full model (ours)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1.  Building blocks
# ─────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU  (standard encoder block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsample → concat skip → two ConvBnRelu → optional MC Dropout.

    The skip connection is already gated by the HANDAttentionGate
    before being passed here, so this block is standard.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                           kernel_size=2, stride=2)
        self.conv1 = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.dropout = nn.Dropout2d(p=dropout_rate)   # spatial dropout

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch from stride/padding
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)       # active at BOTH train and inference
        return x


# ─────────────────────────────────────────────────────────────
# 2.  HAND Attention Gate  (the physics prior)
# ─────────────────────────────────────────────────────────────

class HANDAttentionGate(nn.Module):
    """
    Computes a spatial attention map α(x,y) ∈ [0,1] from two signals:
      g  — gating signal from the decoder (learned, semantic)
      x  — encoder skip feature map
      h  — HAND values (physics prior, metres above nearest drainage)

    The gate equation:
        ψ = ReLU( W_g * g  +  W_x * x  +  W_h * h_proj  +  bias )
        α = Sigmoid( W_ψ * ψ )
        output = α ⊙ x

    Physical interpretation:
      - Low  HAND (near river) → h_proj pushes ψ positive → α → 1
        → decoder sees the full feature map → flood can be detected
      - High HAND (hillside)   → h_proj pushes ψ negative → α → 0
        → decoder feature suppressed → false positives eliminated

    This is NOT just appending HAND as an extra band.
    HAND modulates information flow inside the network.
    """

    def __init__(self, gate_ch: int, feat_ch: int, inter_ch: int):
        """
        Args:
            gate_ch:  channels of the gating signal g (from decoder)
            feat_ch:  channels of the skip feature x (from encoder)
            inter_ch: intermediate channels (typically feat_ch // 2)
        """
        super().__init__()

        # Project each signal to inter_ch
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(feat_ch, inter_ch, kernel_size=1, bias=False),
        )
        # HAND is 1 channel — project to inter_ch
        # Learnable transform: the network decides how to use HAND
        self.W_h = nn.Sequential(
            nn.Conv2d(1, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # Final attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor,
                hand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g    : (B, gate_ch, H/2, W/2) — decoder gating signal (lower res)
            x    : (B, feat_ch, H,   W  ) — encoder skip features  (higher res)
            hand : (B, 1,       H,   W  ) — raw HAND values (metres)

        Returns:
            attended_x : (B, feat_ch, H, W) — gated skip features
            alpha      : (B, 1,       H, W) — attention map (for visualisation)
        """
        # Upsample g to match x spatial resolution
        g_up = F.interpolate(self.W_g(g), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)
        x_proj = self.W_x(x)

        # Normalise HAND: invert so low HAND → high attention (flood-prone)
        # Exponential decay: α(h) = exp(-h / 50)
        #   h=0 m  (stream)   → 1.00  (full attention — flood likely)
        #   h=50 m            → 0.37
        #   h=150 m           → 0.05
        #   h=500 m (Andes)   → ≈0    (no attention — flood impossible)
        # This avoids the hard 100 m clamp that misrepresents Andean terrain
        # in the Bolivia OOD test set and is differentiable everywhere.
        hand_norm = torch.exp(-hand.clamp(min=0.0) / 50.0)  # (B,1,H,W) ∈ (0,1]
        h_proj = self.W_h(hand_norm)

        # Combine and compute attention
        psi_input = self.relu(g_up + x_proj + h_proj)
        alpha = self.psi(psi_input)              # (B, 1, H, W)

        attended_x = alpha * x                   # element-wise gate
        return attended_x, alpha


# ─────────────────────────────────────────────────────────────
# 3.  Siamese Encoder  (ResNet-34, weight-shared)
# ─────────────────────────────────────────────────────────────

class SiameseEncoder(nn.Module):
    """
    Dual-branch ResNet-34 encoder with shared weights.

    Input:  pre-event  (B, 2, H, W) — VV_pre,  VH_pre
            post-event (B, 2, H, W) — VV_post, VH_post

    The first conv layer is rebuilt to accept 2-channel input
    (SAR only has VV + VH, not 3 RGB channels).

    Two encoding modes (controlled by ``diff_mode``):

    diff_mode=False  [Variants A/B/C/D — backward-compatible default]
        Returns post-event features only.  Pre is ignored.
        NOTE: the original design intended difference encoding but the
        implementation was accidentally written as post-only.  Variants
        A/B/C/D were all trained in this mode and their checkpoints
        must be evaluated with diff_mode=False.

    diff_mode=True   [Variant E]
        True Siamese change-detection encoding:
            diff_i = post_features_i − pre_features_i
        The decoder sees the bi-temporal CHANGE at every pyramid level.
        This is theoretically motivated for flood detection:
        flooding ↔ large negative SAR backscatter change (specular return
        from open water replaces rougher land surface).
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 2,
                 diff_mode: bool = False):
        """
        Args:
            pretrained:  use ImageNet-pretrained ResNet-34 weights
            in_channels: number of input channels per branch.
                         2 for standard SAR (VV+VH), 3 for Variant B (VV+VH+HAND).
            diff_mode:   if True, return (post_feats − pre_feats) per level
                         (Variant E).  If False (default), return post_feats
                         only (Variants A/B/C/D — backward-compatible).
        """
        super().__init__()
        self.diff_mode = diff_mode

        # Load ResNet-34 backbone
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet34(weights=None)

        # Rebuild stem conv to accept in_channels instead of 3-channel RGB.
        # Average the 3 RGB weight kernels → 1 channel, then repeat.
        # This preserves pretrained statistics better than random init.
        orig_weight = backbone.conv1.weight.data          # (64, 3, 7, 7)
        avg_weight  = orig_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new_weight  = avg_weight.repeat(1, in_channels, 1, 1)  # (64, C, 7, 7)

        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.conv1.weight = nn.Parameter(new_weight)

        # Extract encoder stages (feature pyramid levels)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu)        # → (B,  64, H/2,  W/2)
        self.pool   = backbone.maxpool                    # → (B,  64, H/4,  W/4)
        self.layer1 = backbone.layer1                     # → (B,  64, H/4,  W/4)
        self.layer2 = backbone.layer2                     # → (B, 128, H/8,  W/8)
        self.layer3 = backbone.layer3                     # → (B, 256, H/16, W/16)
        self.layer4 = backbone.layer4                     # → (B, 512, H/32, W/32)

    def encode_single(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward one branch. Returns feature list [s0,s1,s2,s3,s4]."""
        s0 = self.stem(x)           # (B,  64, H/2,  W/2)
        s0p = self.pool(s0)
        s1 = self.layer1(s0p)       # (B,  64, H/4,  W/4)
        s2 = self.layer2(s1)        # (B, 128, H/8,  W/8)
        s3 = self.layer3(s2)        # (B, 256, H/16, W/16)
        s4 = self.layer4(s3)        # (B, 512, H/32, W/32)
        return [s0, s1, s2, s3, s4]

    def forward(self, pre: torch.Tensor,
                post: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns a 5-level feature pyramid: [f0, f1, f2, f3, f4]

        diff_mode=False (Variants A/B/C/D):
            Returns post-event features only.  Pre is available in the
            input tensor but is discarded here.  Existing checkpoints for
            A/B/C/D are valid under this mode.

        diff_mode=True (Variant E):
            Encodes both branches with shared weights and returns the
            element-wise difference at each pyramid level:
                f_i = post_feats[i] - pre_feats[i]
            Positive values → backscatter INCREASE (rare in floods).
            Negative values → backscatter DECREASE (specular return from
            open water → strong flood signal).
            The decoder + HAND gate then operate on change features.
        """
        if self.diff_mode:
            pre_feats  = self.encode_single(pre)
            post_feats = self.encode_single(post)
            diff_feats = [p - q for p, q in zip(post_feats, pre_feats)]
            return diff_feats
        else:
            # Backward-compatible: post-only encoding (Variants A/B/C/D)
            post_feats = self.encode_single(post)
            return post_feats


# ─────────────────────────────────────────────────────────────
# 4.  Full model
# ─────────────────────────────────────────────────────────────

class FloodSegmentationModel(nn.Module):
    """
    Full model combining:
      - Siamese ResNet-34 encoder
      - HAND-gated attention at each skip connection
      - Decoder with MC Dropout (active at train + inference)
      - Optional population band (appended to bottleneck)

    Ablation config:
      use_hand_gate (bool): if False, skip connections are not gated
      hand_as_band  (bool): if True, HAND is concatenated as input band
                            (only meaningful when use_hand_gate=False)
      dropout_rate  (float): set to 0.0 to disable MC Dropout
      diff_mode     (bool): if True, use true Siamese difference encoding
                            (post_feats − pre_feats at each pyramid level).
                            False (default) uses post-only for A/B/C/D compat.

    Input tensor channel layout (from 02_dataset.py):
      [0] VV_pre
      [1] VH_pre
      [2] VV_post
      [3] VH_post
      [4] VV_VH_ratio  ← present in the 6-band data tensor but NOT fed to the
                         encoder in any current variant; available for future
                         ablation (e.g. concatenating ratio to post branch)
      [5] HAND         ← z-score normalised in the tensor; _prepare_hand()
                         denormalises to metres before passing to the gate

    Encoder input per variant:
      Variant A (SAR only):   post=[ch2,ch3] only (diff_mode=False)  — 2-ch
      Variant B (HAND band):  post=[ch2,ch3,ch5] only (diff_mode=False) — 3-ch
      Variant C (HAND gate):  pre=[ch0,ch1]  post=[ch2,ch3]; ch5→gate
      Variant D (full model): same as C + MC Dropout active at inference

    Note: WorldPop is NOT a model input. It is used only in 06_exposure.py
    post-prediction for population exposure estimation.
    """

    def __init__(
        self,
        pretrained:    bool  = True,
        use_hand_gate: bool  = True,    # Variant C/D/E vs A/B
        hand_as_band:  bool  = False,   # Variant B only
        dropout_rate:  float = 0.3,     # 0.0 = no MC Dropout
        diff_mode:     bool  = False,   # Variant E: true Siamese difference
    ):
        super().__init__()

        self.use_hand_gate = use_hand_gate
        self.hand_as_band  = hand_as_band
        self.dropout_rate  = dropout_rate
        self.diff_mode     = diff_mode

        # ── Encoder ─────────────────────────────────────────
        # Variant B: HAND concatenated onto SAR → 3-channel per branch
        encoder_in_ch = 3 if (hand_as_band and not use_hand_gate) else 2
        self.encoder  = SiameseEncoder(
            pretrained  = pretrained,
            in_channels = encoder_in_ch,
            diff_mode   = diff_mode,
        )

        # ── HAND gates (one per decoder level) ──────────────
        # Only created if use_hand_gate=True
        if use_hand_gate:
            # gate_ch = decoder channels feeding from below
            # feat_ch = encoder skip channels at this level
            self.gate4 = HANDAttentionGate(512, 256, 128)  # skip from layer3
            self.gate3 = HANDAttentionGate(256, 128,  64)  # skip from layer2
            self.gate2 = HANDAttentionGate(128,  64,  32)  # skip from layer1
            self.gate1 = HANDAttentionGate( 64,  64,  32)  # skip from stem

        # ── Decoder blocks ───────────────────────────────────
        self.dec4 = DecoderBlock(512, 256, 256, dropout_rate)
        self.dec3 = DecoderBlock(256, 128, 128, dropout_rate)
        self.dec2 = DecoderBlock(128,  64,  64, dropout_rate)
        self.dec1 = DecoderBlock( 64,  64,  32, dropout_rate)

        # ── Output head ──────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # back to H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # back to H
            # No sigmoid here — use BCEWithLogitsLoss for numerical stability
            # Apply sigmoid manually at inference for probabilities
        )

    # ── MC Dropout control ───────────────────────────────────

    def enable_dropout(self, include_bn: bool = False) -> None:
        """
        Call this before MC Dropout inference to ensure Dropout2d layers
        are active even when the rest of the model is in eval mode.

        Args:
            include_bn: If True, also set all BatchNorm2d layers to train mode.
                        This makes batch statistics stochastic across MC passes
                        (Teye et al. 2018 — "Bayesian Uncertainty Estimation for
                        Batch Normalised Deep Networks").  With include_bn=True
                        the running mean/var are ignored; each forward pass uses
                        the current-batch statistics instead, propagating extra
                        variance through the full encoder + decoder.

                        Expected effect on Variant D:
                          include_bn=False  →  mean_variance ≈ 0.0004  (current)
                          include_bn=True   →  mean_variance ≈ 0.01–0.05 (10–100×)

                        Warning: requires batch_size ≥ 4 per MC pass for stable
                        BN statistics.  Evaluate with the same batch size used
                        during training (8).  Not recommended with batch_size=1.

                        Recommended workflow on DKUCC:
                            model.eval()
                            model.enable_dropout(include_bn=True)
                            # run 05_uncertainty.py with --include_bn flag

        Usage:
            model.eval()
            model.enable_dropout()              # standard MC Dropout
            model.enable_dropout(include_bn=True)  # BN-stochastic MC
            with torch.no_grad():
                for t in range(T):
                    logits = model(batch)
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout2d):
                module.train()
            if include_bn and isinstance(module, nn.BatchNorm2d):
                module.train()   # use batch stats instead of running stats

    # ── HAND preparation ─────────────────────────────────────

    def _prepare_hand(self, x: torch.Tensor,
                      target_size: tuple) -> torch.Tensor:
        """
        Extracts HAND channel from input tensor, denormalises from z-score
        to approximate raw metres, and resizes to target_size.

        WHY the denormalisation is necessary
        ─────────────────────────────────────
        Channel 5 in batch["image"] holds z-score normalised HAND:
            z = (HAND_metres − 9.346) / 28.330

        HANDAttentionGate applies:
            gate = exp(−h / 50.0)

        This decay constant (50 m) is calibrated in metres.
        Without denormalisation h is a z-score and the gate collapses:
            HAND= 50 m → z=+1.43 → gate=exp(−1.43/50)=0.972  [should be 0.37]
            HAND=500 m → z=+17.3 → gate=exp(−17.3/50)=0.707  [should be ≈0.00]

        With denormalisation:
            HAND= 50 m → gate = exp(−50/50)  = 0.368  ✓
            HAND=500 m → gate = exp(−500/50) ≈ 0.000  ✓  (Andes suppressed)

        Constants from norm_stats.json (train split, 364 chips):
            HAND mean = 9.346 m,  HAND std = 28.330 m
        """
        hand_z = x[:, 5:6, :, :]   # (B, 1, H, W) — z-score normalised
        # Denormalise to approximate raw HAND in metres so the physics gate
        # exp(-h / 50) operates on the correct physical scale.
        _HAND_MEAN: float = 9.346   # metres — from norm_stats.json
        _HAND_STD:  float = 28.330  # metres — from norm_stats.json
        hand = hand_z * _HAND_STD + _HAND_MEAN   # (B, 1, H, W) ≈ raw metres
        if hand.shape[-2:] != target_size:
            hand = F.interpolate(hand, size=target_size,
                                 mode="bilinear", align_corners=False)
        return hand

    # ── Forward pass ─────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) input tensor
               Channels: [VV_pre, VH_pre, VV_post, VH_post,
                          VV_VH_ratio, HAND]

        Returns:
            logits: (B, 1, H, W) — raw logits (before sigmoid)
                    Apply sigmoid for flood probability maps.
        """
        B, _, H, W = x.shape

        # ── Split input bands ────────────────────────────────
        pre  = x[:, 0:2, :, :]    # VV_pre,  VH_pre
        post = x[:, 2:4, :, :]    # VV_post, VH_post

        # If HAND as band: concatenate HAND onto pre/post inputs
        if self.hand_as_band and not self.use_hand_gate:
            hand_band = x[:, 5:6, :, :]
            pre  = torch.cat([pre,  hand_band], dim=1)   # 3-channel
            post = torch.cat([post, hand_band], dim=1)   # 3-channel
            # Note: encoder was built for 2-channel; this path requires
            # retraining with hand_as_band=True from scratch (different model)

        # ── Encoder: feature difference pyramid ─────────────
        # diffs = [diff0, diff1, diff2, diff3, diff4]
        # diff_i shape: see SiameseEncoder.encode_single comments
        diffs = self.encoder(pre, post)
        d0, d1, d2, d3, d4 = diffs   # d4 is bottleneck

        # ── Decoder with optional HAND gates ────────────────
        # At each level: gate the encoder skip → decoder block

        if self.use_hand_gate:
            # Gate 4: d4 (bottleneck) gates d3 skip
            hand_d3 = self._prepare_hand(x, d3.shape[-2:])
            d3_gated, alpha3 = self.gate4(d4, d3, hand_d3)
            out = self.dec4(d4, d3_gated)

            # Gate 3: decoder out gates d2 skip
            hand_d2 = self._prepare_hand(x, d2.shape[-2:])
            d2_gated, alpha2 = self.gate3(out, d2, hand_d2)
            out = self.dec3(out, d2_gated)

            # Gate 2: decoder out gates d1 skip
            hand_d1 = self._prepare_hand(x, d1.shape[-2:])
            d1_gated, alpha1 = self.gate2(out, d1, hand_d1)
            out = self.dec2(out, d1_gated)

            # Gate 1: decoder out gates d0 skip
            hand_d0 = self._prepare_hand(x, d0.shape[-2:])
            d0_gated, alpha0 = self.gate1(out, d0, hand_d0)
            out = self.dec1(out, d0_gated)

        else:
            # No gating — standard skip connections
            out = self.dec4(d4, d3)
            out = self.dec3(out, d2)
            out = self.dec2(out, d1)
            out = self.dec1(out, d0)

        # ── Output ───────────────────────────────────────────
        logits = self.output_head(out)    # (B, 1, H, W)

        # Ensure output matches input spatial resolution exactly
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W),
                                   mode="bilinear", align_corners=False)

        return logits   # raw logits — caller applies sigmoid

    # ── Gate visualisation forward pass ──────────────────────

    def forward_with_gates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[list[np.ndarray]]]:
        """
        Like forward(), but also returns the 4 HAND gate attention maps
        upsampled to input resolution for visualisation.

        Only meaningful for Variants C and D (use_hand_gate=True).
        For Variants A and B returns (logits, None).

        Args:
            x: (B, 6, H, W) input tensor

        Returns:
            logits    : (B, 1, H, W) raw logits
            gate_maps : list of 4 numpy arrays, each (B, H, W) float32 ∈ [0,1]
                        ordered finest→coarsest [alpha0, alpha1, alpha2, alpha3]
                        or None if use_hand_gate=False
        """
        if not self.use_hand_gate:
            return self.forward(x), None

        B, _, H, W = x.shape

        pre  = x[:, 0:2, :, :]
        post = x[:, 2:4, :, :]
        diffs = self.encoder(pre, post)
        d0, d1, d2, d3, d4 = diffs

        # Gate 4: bottleneck → d3 skip
        hand_d3 = self._prepare_hand(x, d3.shape[-2:])
        d3_gated, alpha3 = self.gate4(d4, d3, hand_d3)
        out = self.dec4(d4, d3_gated)

        # Gate 3: decoder → d2 skip
        hand_d2 = self._prepare_hand(x, d2.shape[-2:])
        d2_gated, alpha2 = self.gate3(out, d2, hand_d2)
        out = self.dec3(out, d2_gated)

        # Gate 2: decoder → d1 skip
        hand_d1 = self._prepare_hand(x, d1.shape[-2:])
        d1_gated, alpha1 = self.gate2(out, d1, hand_d1)
        out = self.dec2(out, d1_gated)

        # Gate 1: decoder → d0 skip  (finest, H/2)
        hand_d0 = self._prepare_hand(x, d0.shape[-2:])
        d0_gated, alpha0 = self.gate1(out, d0, hand_d0)
        out = self.dec1(out, d0_gated)

        logits = self.output_head(out)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W),
                                   mode="bilinear", align_corners=False)

        # Upsample all gate maps to input (H, W) for direct visual comparison.
        # Return as CPU numpy arrays (no grad) so plotting code needs no torch.
        gate_maps: list[np.ndarray] = []
        for alpha in [alpha0, alpha1, alpha2, alpha3]:
            alpha_up = F.interpolate(alpha, size=(H, W),
                                     mode="bilinear", align_corners=False)
            # Shape: (B, H, W)
            gate_maps.append(
                alpha_up.squeeze(1).detach().cpu().float().numpy()
            )

        return logits, gate_maps

    # ── Parameter count ──────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        encoder   = sum(p.numel() for p in self.encoder.parameters())
        return {
            "total":     total,
            "trainable": trainable,
            "encoder":   encoder,
            "decoder":   total - encoder,
        }


# ─────────────────────────────────────────────────────────────
# 5.  Loss function
# ─────────────────────────────────────────────────────────────

class FloodLoss(nn.Module):
    """
    Combined BCE + Dice loss for flood segmentation.

    BCE handles pixel-level calibration.
    Dice handles class imbalance (flood pixels are rare, ~3-5%).

    alpha controls the mix:
      alpha=0.5 → equal BCE and Dice  (recommended default)
      alpha=1.0 → BCE only
      alpha=0.0 → Dice only

    Ignores pixels with label == -1 (invalid/masked).
    """

    def __init__(self, pos_weight: float = 10.0, alpha: float = 0.5,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha     = alpha
        self.smooth    = smooth
        self.bce       = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def dice_loss(self, logits: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(-1)
        targets = targets.view(-1).float()
        intersection = (probs * targets).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return dice

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W) raw model output
            labels : (B, H, W)    int64, values 0/1, -1=ignore
        """
        # Reshape logits to (B, H, W)
        logits_sq = logits.squeeze(1)

        # Build ignore mask
        valid = labels != -1
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device,
                                requires_grad=True)

        logits_valid = logits_sq[valid]
        labels_valid = labels[valid].float()

        # Move pos_weight to correct device
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce_loss  = self.bce(logits_valid, labels_valid)
        dice_loss = self.dice_loss(logits_valid.unsqueeze(0),
                                   labels_valid.unsqueeze(0))

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


# ─────────────────────────────────────────────────────────────
# 6.  Ablation model factory
# ─────────────────────────────────────────────────────────────

def build_model(variant: str = "D",
                pretrained: bool = True,
                dropout_rate: Optional[float] = None) -> FloodSegmentationModel:
    """
    Factory for the ablation variants.

    Variant A — SAR only baseline (no HAND, no UQ)
    Variant B — HAND as extra input band (no gate, no UQ)
    Variant C — HAND gate, no MC Dropout
    Variant D — Full model: HAND gate + MC Dropout  (default dropout=0.3)
    Variant E — True Siamese diff + HAND gate + MC Dropout

    Args:
        variant:      One of A / B / C / D / E.
        pretrained:   Load ImageNet weights for ResNet-34 encoder.
        dropout_rate: Optional override for the variant's default dropout rate.
                      Pass e.g. 0.5 to train D-prime without changing the variant
                      letter.  None (default) uses the variant's built-in value.
                      Stored in checkpoint config so eval/uncertainty scripts can
                      reconstruct the exact model that was trained.

    Usage:
        model = build_model("D")                        # standard D, dropout=0.3
        model = build_model("D", dropout_rate=0.5)      # D-prime, dropout=0.5
        print(model.count_parameters())
    """
    configs = {
        # Variants A-D: diff_mode=False (post-only; backward-compatible)
        "A": dict(use_hand_gate=False, hand_as_band=False, dropout_rate=0.0, diff_mode=False),
        "B": dict(use_hand_gate=False, hand_as_band=True,  dropout_rate=0.0, diff_mode=False),
        "C": dict(use_hand_gate=True,  hand_as_band=False, dropout_rate=0.0, diff_mode=False),
        "D": dict(use_hand_gate=True,  hand_as_band=False, dropout_rate=0.3, diff_mode=False),
        # Variant E: TRUE Siamese difference + HAND gate + MC Dropout
        # Fixes the diff_mode=False bug in A-D by enabling genuine change-detection
        # encoding: decoder features = post_feats[i] − pre_feats[i] at each level.
        # Must be trained from scratch; checkpoints are NOT compatible with D.
        "E": dict(use_hand_gate=True,  hand_as_band=False, dropout_rate=0.3, diff_mode=True),
    }
    if variant not in configs:
        raise ValueError(f"Variant must be A/B/C/D/E, got '{variant}'")

    cfg = dict(configs[variant])                # copy — do not mutate the table
    if dropout_rate is not None:
        cfg["dropout_rate"] = dropout_rate      # apply caller override

    model = FloodSegmentationModel(pretrained=pretrained, **cfg)
    print(f"Built Variant {variant}: {model.count_parameters()}")
    return model


# ─────────────────────────────────────────────────────────────
# 7.  Sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test all variants — 6-band input (no pop_log)
    batch = torch.randn(2, 6, 256, 256).to(device)

    for variant in ["A", "B", "C", "D", "E"]:
        model = build_model(variant, pretrained=False).to(device)
        model.train()

        logits = model(batch)
        print(f"  Variant {variant} output: {logits.shape}"
              f"  min={logits.min():.2f}  max={logits.max():.2f}")

        assert logits.shape == (2, 1, 256, 256), \
            f"Expected (2,1,256,256), got {logits.shape}"

    # Test MC Dropout inference (Variant D)
    print("\nMC Dropout test (T=5 passes):")
    model_d = build_model("D", pretrained=False).to(device)
    model_d.eval()
    model_d.enable_dropout()   # keep dropout active

    preds = []
    with torch.no_grad():
        for t in range(5):
            logits = model_d(batch)
            preds.append(torch.sigmoid(logits))

    preds  = torch.stack(preds)          # (T, B, 1, H, W)
    mean   = preds.mean(0)
    var    = preds.var(0)
    print(f"  Predictive mean  shape: {mean.shape}")
    print(f"  Predictive var   shape: {var.shape}")
    print(f"  Mean variance (uncertainty): {var.mean():.4f}")
    print(f"  Variance range: [{var.min():.4f}, {var.max():.4f}]")

    # Test loss
    print("\nLoss test:")
    labels = torch.randint(0, 2, (2, 256, 256)).to(device)
    labels[0, :10, :10] = -1   # inject some ignore pixels
    criterion = FloodLoss(pos_weight=10.0, alpha=0.5)
    model_d.train()
    logits = model_d(batch)
    loss = criterion(logits, labels)
    loss.backward()
    print(f"  Loss value: {loss.item():.4f}  (backward OK)")

    print("\nAll checks passed. Ready for Phase 3 (04_train.py).")
