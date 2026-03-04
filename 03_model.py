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

        # Normalise HAND: invert so low HAND → high value (more attention)
        # hand is in metres; clamp to [0, 100] before normalisation
        hand_norm = 1.0 - (hand.clamp(0, 100) / 100.0)  # 0=hilltop, 1=drainage
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

    Outputs feature pyramid from both branches, then returns
    the element-wise DIFFERENCE at each scale:
        diff_i = post_features_i - pre_features_i

    Difference encoding is theoretically motivated for change detection:
    flooding = what changed between pre and post.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet-34 backbone
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet34(weights=None)

        # Rebuild stem conv: 3-channel RGB → 2-channel SAR (VV + VH)
        # We average the 3 RGB weight kernels → 2-channel approximation
        # This preserves pretrained statistics better than random init
        orig_weight = backbone.conv1.weight.data   # (64, 3, 7, 7)
        new_weight  = orig_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new_weight  = new_weight.repeat(1, 2, 1, 1)          # (64, 2, 7, 7)

        backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
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
        Returns list of feature differences: [diff0, diff1, diff2, diff3, diff4]
        Each diff_i = post_features_i - pre_features_i
        """
        pre_feats  = self.encode_single(pre)
        post_feats = self.encode_single(post)
        diffs = [p - q for p, q in zip(post_feats, pre_feats)]
        return diffs


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

    Input tensor channel layout (from 02_dataset.py):
      [0] VV_pre
      [1] VH_pre
      [2] VV_post
      [3] VH_post
      [4] VV_VH_ratio  (used as extra feature when hand_as_band=False)
      [5] HAND         (used by gate OR as band, depending on config)
      [6] pop_log      (appended to bottleneck)
    """

    def __init__(
        self,
        pretrained:    bool  = True,
        use_hand_gate: bool  = True,    # Variant C/D vs A/B
        hand_as_band:  bool  = False,   # Variant B only
        use_pop:       bool  = True,    # include WorldPop band
        dropout_rate:  float = 0.3,     # 0.0 = no MC Dropout
    ):
        super().__init__()

        self.use_hand_gate = use_hand_gate
        self.hand_as_band  = hand_as_band
        self.use_pop       = use_pop
        self.dropout_rate  = dropout_rate

        # ── Encoder ─────────────────────────────────────────
        self.encoder = SiameseEncoder(pretrained=pretrained)

        # ── Bottleneck: optional population injection ────────
        # Population density (pop_log) is appended at the bottleneck
        # because it's a coarse signal (100m → resampled) that should
        # influence high-level decisions, not low-level features
        bottleneck_extra = 1 if use_pop else 0
        self.bottleneck_proj = nn.Sequential(
            ConvBnRelu(512 + bottleneck_extra, 512),
            ConvBnRelu(512, 512),
        ) if use_pop else nn.Identity()

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

    def enable_dropout(self):
        """
        Call this before MC Dropout inference to ensure Dropout2d layers
        are active even when the rest of the model is in eval mode.

        Usage:
            model.eval()
            model.enable_dropout()
            with torch.no_grad():
                for t in range(T):
                    logits = model(batch)
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout2d):
                module.train()

    # ── HAND preparation ─────────────────────────────────────

    def _prepare_hand(self, x: torch.Tensor,
                      target_size: tuple) -> torch.Tensor:
        """
        Extracts HAND channel from input tensor and resizes to target_size.
        HAND is channel index 5 in our 7-band input stack.
        """
        hand = x[:, 5:6, :, :]   # (B, 1, H, W)
        if hand.shape[-2:] != target_size:
            hand = F.interpolate(hand, size=target_size,
                                 mode="bilinear", align_corners=False)
        return hand

    # ── Forward pass ─────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, H, W) input tensor
               Channels: [VV_pre, VH_pre, VV_post, VH_post,
                          VV_VH_ratio, HAND, pop_log]

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

        pop  = x[:, 6:7, :, :]    # population log-density

        # ── Encoder: feature difference pyramid ─────────────
        # diffs = [diff0, diff1, diff2, diff3, diff4]
        # diff_i shape: see SiameseEncoder.encode_single comments
        diffs = self.encoder(pre, post)
        d0, d1, d2, d3, d4 = diffs   # d4 is bottleneck

        # ── Bottleneck: inject population ───────────────────
        if self.use_pop:
            pop_resized = F.interpolate(
                pop, size=d4.shape[-2:], mode="bilinear", align_corners=False
            )
            d4 = torch.cat([d4, pop_resized], dim=1)   # (B, 513, H/32, W/32)
            d4 = self.bottleneck_proj(d4)               # back to (B, 512, ...)

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
                pretrained: bool = True) -> FloodSegmentationModel:
    """
    Factory for the 4 ablation variants.

    Variant A — SAR only baseline (no HAND, no UQ)
    Variant B — HAND as extra input band (no gate, no UQ)
    Variant C — HAND gate, no MC Dropout
    Variant D — Full model: HAND gate + MC Dropout  ← paper submission

    Usage:
        model = build_model("D")
        print(model.count_parameters())
    """
    configs = {
        "A": dict(use_hand_gate=False, hand_as_band=False,
                  use_pop=False, dropout_rate=0.0),
        "B": dict(use_hand_gate=False, hand_as_band=True,
                  use_pop=False, dropout_rate=0.0),
        "C": dict(use_hand_gate=True,  hand_as_band=False,
                  use_pop=True,  dropout_rate=0.0),
        "D": dict(use_hand_gate=True,  hand_as_band=False,
                  use_pop=True,  dropout_rate=0.3),
    }
    if variant not in configs:
        raise ValueError(f"Variant must be A/B/C/D, got '{variant}'")

    model = FloodSegmentationModel(pretrained=pretrained, **configs[variant])
    print(f"Built Variant {variant}: {model.count_parameters()}")
    return model


# ─────────────────────────────────────────────────────────────
# 7.  Sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test all 4 variants
    batch = torch.randn(2, 7, 256, 256).to(device)

    for variant in ["A", "B", "C", "D"]:
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
