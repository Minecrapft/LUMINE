"""
LUMINE White Reference Normalization

Samples the white plastic substrate of the dipstick (e.g. from the bare handle
or spaces around pads) to compute per-channel gains, compensating for
warm/cool ambient illumination before CIE L*a*b* conversion.

Limitations:
- Cannot correct non-linear camera tone curves, clipped channels (saturated highlights),
  or severe non-uniform lighting / shadows across the strip length.
- Assumes the plastic substrate is nominally neutral white / light cream (D65).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WhiteReferenceNormalizer:
    """
    Normalizes measured pad RGB colors using the strip's white substrate reference.
    """

    def __init__(self, target_white: Tuple[float, float, float] = (245.0, 245.0, 245.0),
                 min_gain: float = 0.5, max_gain: float = 2.0):
        self.target_white = np.array(target_white, dtype=np.float32)
        self.min_gain = min_gain
        self.max_gain = max_gain

    def sample_substrate(self, rectified_strip: np.ndarray) -> np.ndarray:
        """
        Samples the white plastic substrate from the canonical horizontal rectified strip.
        In canonical orientation, the handle section occupies the right 18-25% of the strip.
        """
        if rectified_strip is None or rectified_strip.size == 0:
            return np.array([240.0, 240.0, 240.0], dtype=np.float32)

        h, w = rectified_strip.shape[:2]
        # Handle region: rightmost 18% of strip length, middle 60% vertical
        hx1 = int(w * 0.82)
        hx2 = int(w * 0.98)
        hy1 = int(h * 0.20)
        hy2 = int(h * 0.80)

        handle_roi = rectified_strip[hy1:hy2, hx1:hx2]
        if handle_roi.size == 0:
            # Fallback to borders
            handle_roi = rectified_strip[:int(h * 0.15), :]

        # Mean BGR -> RGB
        mean_bgr = np.mean(handle_roi, axis=(0, 1))
        measured_white_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)
        return measured_white_rgb

    def compute_gains(self, measured_white: np.ndarray) -> np.ndarray:
        """
        Computes per-channel gain factors [gain_R, gain_G, gain_B] to map measured
        substrate white to target white.
        """
        measured = np.maximum(measured_white, 1.0)
        gains = self.target_white / measured
        clamped_gains = np.clip(gains, self.min_gain, self.max_gain)
        return clamped_gains

    def normalize_color(self, rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """
        Applies white reference gains to measured pad RGB.
        """
        rgb_arr = np.array(rgb, dtype=np.float32)
        normalized = np.clip(rgb_arr * gains, 0.0, 255.0)
        return normalized

    def normalize_strip_pads(
        self,
        rectified_strip: np.ndarray,
        pads: list
    ) -> Tuple[list, Dict[str, Any]]:
        """
        Samples substrate white from the rectified strip, computes gains,
        and applies normalization to all pads.
        """
        measured_white = self.sample_substrate(rectified_strip)
        gains = self.compute_gains(measured_white)

        normalized_pads = []
        for p in pads:
            p_copy = dict(p)
            raw_rgb = p.get('color', [128, 128, 128])
            norm_rgb = self.normalize_color(raw_rgb, gains)
            norm_int = [int(round(c)) for c in norm_rgb]
            p_copy['color_raw'] = raw_rgb
            p_copy['color'] = norm_int
            p_copy['color_normalized'] = [c / 255.0 for c in norm_int]
            normalized_pads.append(p_copy)

        meta = {
            'measured_white_rgb': [float(c) for c in measured_white],
            'gains_rgb': [float(g) for g in gains],
            'target_white_rgb': [float(c) for c in self.target_white]
        }
        return normalized_pads, meta
