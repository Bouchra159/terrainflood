"""
Trust Mask — Standalone Module
================================
File: trust_mask.py

Computes and applies trust masks from MC Dropout predictive variance.
Used by 06_exposure.py and eval.py.

A trust mask marks pixels where the model is confident enough to be
included in downstream analyses (population exposure, flood mapping).

Functions:
  compute_trust_mask(variance, threshold)  → bool array
  apply_trust_mask(flood_prob, trust_mask) → masked probability array
  summarise_trust_mask(trust_mask, label)  → coverage stats dict
"""

import numpy as np


def compute_trust_mask(
    variance:  np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Binary trust mask from MC Dropout predictive variance.

    Pixels with variance <= threshold are considered confident (trusted).
    Variance of 0.05 corresponds to ±0.22 std dev in probability —
    the model disagrees significantly across MC passes above this level.

    Args:
        variance:  (H, W) or (B, H, W) predictive variance from MC Dropout
        threshold: max allowed variance for a trusted pixel (default 0.05)

    Returns:
        trust_mask: same shape as variance, dtype bool
                    True  = confident prediction (trust this pixel)
                    False = uncertain (exclude from downstream analysis)
    """
    return variance <= threshold


def apply_trust_mask(
    flood_prob:  np.ndarray,
    trust_mask:  np.ndarray,
    fill_value:  float = float("nan"),
) -> np.ndarray:
    """
    Applies a trust mask to a flood probability map.

    Untrusted pixels are set to fill_value (NaN by default).

    Args:
        flood_prob:  (H, W) or (B, H, W) flood probability in [0, 1]
        trust_mask:  same shape, bool — True = trusted pixel
        fill_value:  value to assign to untrusted pixels

    Returns:
        masked_prob: same shape as flood_prob, with untrusted pixels = fill_value
    """
    masked = flood_prob.astype(np.float32).copy()
    masked[~trust_mask] = fill_value
    return masked


def summarise_trust_mask(
    trust_mask:   np.ndarray,
    label:        np.ndarray | None = None,
    ignore_value: int = -1,
) -> dict:
    """
    Computes coverage statistics for a trust mask.

    Args:
        trust_mask:   (H, W) bool array from compute_trust_mask()
        label:        optional (H, W) ground truth for flood recall stats
        ignore_value: label value to exclude from stats (-1 = invalid pixels)

    Returns:
        dict with keys:
          total_pixels         : int   — valid pixels counted
          trusted_pixels       : int   — trusted valid pixels
          coverage             : float — trusted / total  (target: > 0.7)
          uncertain_pixels     : int
          trusted_flood_recall : float | None  (requires label)
    """
    if label is not None:
        valid = label != ignore_value
    else:
        valid = np.ones_like(trust_mask, dtype=bool)

    total   = int(valid.sum())
    trusted = int((trust_mask & valid).sum())

    result: dict = {
        "total_pixels":     total,
        "trusted_pixels":   trusted,
        "coverage":         trusted / max(total, 1),
        "uncertain_pixels": total - trusted,
    }

    if label is not None:
        flood_pixels  = (label == 1) & valid
        trusted_flood = int((trust_mask & flood_pixels).sum())
        result["trusted_flood_recall"] = float(trusted_flood) / max(int(flood_pixels.sum()), 1)
    else:
        result["trusted_flood_recall"] = None

    return result
