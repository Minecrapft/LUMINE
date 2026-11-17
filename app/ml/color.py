"""
app/ml/color.py — Colour-space utilities for LUMINE.

All conversions use the CIE D65 illuminant so that ΔE values are
perceptually meaningful.  Requires scikit-image (skimage).

Public API
----------
rgb_to_lab(rgb)         → [L*, a*, b*]
lab_to_rgb(lab)         → [R, G, B]   (uint8, clipped)
delta_e(lab1, lab2)     → float       (CIEDE2000)
rgb_delta_e(rgb1, rgb2) → float       (convenience wrapper)
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Union

# ---------------------------------------------------------------------------
# Import the heavy-lifting from scikit-image.  We do this here so every other
# module only imports from *this* file and scikit-image is listed once.
# ---------------------------------------------------------------------------
try:
    from skimage.color import rgb2lab as _skimage_rgb2lab, lab2rgb as _skimage_lab2rgb
    from skimage.color import deltaE_ciede2000 as _skimage_de2000
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float_rgb(rgb: Sequence) -> np.ndarray:
    """Accept list/array of 3 values (0–255 OR 0.0–1.0) and return float [0,1]."""
    arr = np.asarray(rgb, dtype=float)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: Union[Sequence, np.ndarray]) -> np.ndarray:
    """
    Convert an sRGB triplet to CIE L*a*b* (D65 illuminant).

    Parameters
    ----------
    rgb : array-like of length 3
        Either uint8 in [0, 255] or float in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        [L*, a*, b*] where L* ∈ [0, 100], a*/b* ∈ [-128, 127].
    """
    if not _SKIMAGE_AVAILABLE:
        return _rgb_to_lab_fallback(rgb)

    float_rgb = _as_float_rgb(rgb)
    # skimage expects shape (H, W, 3); we use (1, 1, 3)
    pixel = float_rgb.reshape(1, 1, 3)
    lab_pixel = _skimage_rgb2lab(pixel)
    return lab_pixel[0, 0, :]  # shape (3,)


def lab_to_rgb(lab: Union[Sequence, np.ndarray]) -> np.ndarray:
    """
    Convert CIE L*a*b* back to sRGB uint8 [0, 255].

    Parameters
    ----------
    lab : array-like of length 3
        [L*, a*, b*]

    Returns
    -------
    np.ndarray
        [R, G, B] as uint8.
    """
    if not _SKIMAGE_AVAILABLE:
        return np.array([128, 128, 128], dtype=np.uint8)

    lab_arr = np.asarray(lab, dtype=float).reshape(1, 1, 3)
    rgb_float = _skimage_lab2rgb(lab_arr)
    rgb_uint8 = np.clip(rgb_float[0, 0, :] * 255, 0, 255).astype(np.uint8)
    return rgb_uint8


def delta_e(lab1: Union[Sequence, np.ndarray],
            lab2: Union[Sequence, np.ndarray]) -> float:
    """
    Compute CIEDE2000 colour difference between two Lab values.

    Returns
    -------
    float
        ΔE₀₀ ≥ 0.  Values < 1 are imperceptible; > 10 are clearly different.
    """
    if not _SKIMAGE_AVAILABLE:
        # Euclidean in Lab as rough fallback
        return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))

    arr1 = np.asarray(lab1, dtype=float).reshape(1, 1, 3)
    arr2 = np.asarray(lab2, dtype=float).reshape(1, 1, 3)
    de = _skimage_de2000(arr1, arr2)
    return float(de[0, 0])


def rgb_delta_e(rgb1: Union[Sequence, np.ndarray],
                rgb2: Union[Sequence, np.ndarray]) -> float:
    """
    Compute CIEDE2000 directly from two sRGB triplets (convenience wrapper).
    """
    return delta_e(rgb_to_lab(rgb1), rgb_to_lab(rgb2))


# ---------------------------------------------------------------------------
# Pure-Python fallback (no scikit-image) — approximate sRGB → XYZ → Lab
# Not as accurate as skimage but keeps the app working if skimage is absent.
# ---------------------------------------------------------------------------

def _linearize(channel: float) -> float:
    """sRGB gamma linearisation."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _rgb_to_lab_fallback(rgb: Sequence) -> np.ndarray:
    """Fallback sRGB → Lab using approximate matrix + cube-root transform."""
    r, g, b = (_linearize(v / 255.0) if max(rgb) > 1 else _linearize(float(v))
               for v in rgb)

    # sRGB → XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ → Lab (D65 white: Xn=0.95047, Yn=1.0000, Zn=1.08883)
    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx = f(x / 0.95047)
    fy = f(y / 1.00000)
    fz = f(z / 1.08883)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_star = 200 * (fy - fz)

    return np.array([L, a, b_star], dtype=float)
