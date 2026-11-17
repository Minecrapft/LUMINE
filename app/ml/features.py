"""
app/ml/features.py — Feature extraction for LUMINE reagent pad classification.

Extracts colour features in CIE L*a*b*, HSV, and normalized chromaticity
spaces, along with dispersion and illumination-invariant differences against
the strip's white reference substrate.

FEATURE_VERSION is embedded in model artifacts to ensure compatibility and
prevent silent drift.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence, Union, Optional, List, Dict, Any

from app.ml.color import rgb_to_lab, delta_e

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
FEATURE_VERSION: int = 1

FEATURE_NAMES: List[str] = [
    "lab_L_mean", "lab_a_mean", "lab_b_mean",
    "lab_L_median", "lab_a_median", "lab_b_median",
    "hsv_h_mean", "hsv_s_mean", "hsv_v_mean",
    "hsv_h_median", "hsv_s_median", "hsv_v_median",
    "chrom_r_mean", "chrom_g_mean",
    "chrom_r_median", "chrom_g_median",
    "dispersion",
    "white_diff_L", "white_diff_a", "white_diff_b",
    "white_delta_e",
    "white_ratio_r", "white_ratio_g", "white_ratio_b"
]

DEFAULT_WHITE_RGB = np.array([245.0, 245.0, 245.0], dtype=np.float32)


def extract_pad_features(
    patch_or_rgb: Union[np.ndarray, Sequence[float]],
    white_ref_rgb: Optional[Union[np.ndarray, Sequence[float]]] = None,
    dispersion: Optional[float] = None
) -> np.ndarray:
    """
    Extract a 24-dimensional feature vector for a single reagent pad.

    Parameters
    ----------
    patch_or_rgb : np.ndarray or sequence
        Can be:
        - An image patch array of shape (H, W, 3) in RGB (uint8 or float)
        - A pixel array of shape (N, 3) in RGB
        - A single RGB triplet [R, G, B] (0-255 or 0.0-1.0)
    white_ref_rgb : np.ndarray or sequence, optional
        RGB of the white reference substrate. Default is [245, 245, 245].
    dispersion : float, optional
        Pre-computed dispersion (spatial variance). If None and patch is 2D/3D,
        computed directly from pixel standard deviation.

    Returns
    -------
    np.ndarray
        1D float32 array of shape (24,) matching FEATURE_NAMES.
    """
    # 1. Parse and standardize white reference
    if white_ref_rgb is None:
        white_rgb_255 = DEFAULT_WHITE_RGB.copy()
    else:
        w = np.asarray(white_ref_rgb, dtype=np.float32).ravel()
        if len(w) != 3:
            w = DEFAULT_WHITE_RGB.copy()
        if w.max() <= 1.0 and not np.allclose(w, 0):
            w = w * 255.0
        white_rgb_255 = np.clip(w, 1.0, 255.0)

    white_lab = rgb_to_lab(white_rgb_255)

    # 2. Process input patch / pixels
    arr = np.asarray(patch_or_rgb, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[2] == 3:
        pixels_rgb = arr.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        pixels_rgb = arr
    elif arr.ndim == 1 and len(arr) == 3:
        pixels_rgb = arr.reshape(1, 3)
    else:
        raise ValueError(
            f"Expected patch array of shape (H, W, 3), (N, 3), or length 3. Got shape {arr.shape}"
        )

    # Normalize pixels to [0, 255]
    if pixels_rgb.max() <= 1.0 and not np.allclose(pixels_rgb, 0):
        pixels_rgb_255 = pixels_rgb * 255.0
    else:
        pixels_rgb_255 = pixels_rgb.copy()
    pixels_rgb_255 = np.clip(pixels_rgb_255, 0.0, 255.0)

    # 3. Compute dispersion
    if dispersion is not None:
        pad_dispersion = float(dispersion)
    elif len(pixels_rgb_255) > 1:
        pad_dispersion = float(np.mean(np.std(pixels_rgb_255, axis=0)))
    else:
        pad_dispersion = 0.0

    # 4. RGB statistics
    mean_rgb = np.mean(pixels_rgb_255, axis=0)

    # 5. CIE L*a*b* conversion & statistics
    pix_norm = (pixels_rgb_255 / 255.0).reshape(-1, 1, 3)
    try:
        from skimage.color import rgb2lab as _skimage_rgb2lab
        lab_pixels = _skimage_rgb2lab(pix_norm).reshape(-1, 3)
    except Exception:
        lab_pixels = np.array([rgb_to_lab(p) for p in pixels_rgb_255])

    lab_mean = np.mean(lab_pixels, axis=0)
    lab_median = np.median(lab_pixels, axis=0)

    # 6. HSV conversion & statistics
    u8_pixels = pixels_rgb_255.astype(np.uint8).reshape(-1, 1, 3)
    hsv_pixels = cv2.cvtColor(u8_pixels, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv_mean = np.mean(hsv_pixels, axis=0)
    hsv_median = np.median(hsv_pixels, axis=0)

    # 7. Normalized chromaticity: r = R/(R+G+B), g = G/(R+G+B)
    sum_rgb = np.sum(pixels_rgb_255, axis=1, keepdims=True)
    sum_rgb[sum_rgb < 1e-4] = 1e-4
    chrom_r = pixels_rgb_255[:, 0:1] / sum_rgb
    chrom_g = pixels_rgb_255[:, 1:2] / sum_rgb

    chrom_r_mean   = float(np.mean(chrom_r))
    chrom_g_mean   = float(np.mean(chrom_g))
    chrom_r_median = float(np.median(chrom_r))
    chrom_g_median = float(np.median(chrom_g))

    # 8. White reference relative differences
    white_diff_L   = float(lab_mean[0] - white_lab[0])
    white_diff_a   = float(lab_mean[1] - white_lab[1])
    white_diff_b   = float(lab_mean[2] - white_lab[2])
    white_delta_e  = float(delta_e(lab_mean, white_lab))
    white_ratio_r  = float(mean_rgb[0] / max(1.0, float(white_rgb_255[0])))
    white_ratio_g  = float(mean_rgb[1] / max(1.0, float(white_rgb_255[1])))
    white_ratio_b  = float(mean_rgb[2] / max(1.0, float(white_rgb_255[2])))

    # 9. Assemble feature vector
    feat = [
        float(lab_mean[0]),   float(lab_mean[1]),   float(lab_mean[2]),
        float(lab_median[0]), float(lab_median[1]), float(lab_median[2]),
        float(hsv_mean[0]),   float(hsv_mean[1]),   float(hsv_mean[2]),
        float(hsv_median[0]), float(hsv_median[1]), float(hsv_median[2]),
        chrom_r_mean, chrom_g_mean,
        chrom_r_median, chrom_g_median,
        pad_dispersion,
        white_diff_L, white_diff_a, white_diff_b,
        white_delta_e,
        white_ratio_r, white_ratio_g, white_ratio_b,
    ]

    return np.array(feat, dtype=np.float32)


def extract_features_batch(
    patches_or_rgbs: Sequence[Union[np.ndarray, Sequence[float]]],
    white_ref_rgb: Optional[Union[np.ndarray, Sequence[float]]] = None,
    dispersions: Optional[Sequence[Optional[float]]] = None
) -> np.ndarray:
    """
    Extract features for a batch of pads.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (N, 24).
    """
    feats = []
    for i, item in enumerate(patches_or_rgbs):
        disp = dispersions[i] if dispersions and i < len(dispersions) else None
        feats.append(extract_pad_features(item, white_ref_rgb=white_ref_rgb, dispersion=disp))
    return np.array(feats, dtype=np.float32)
