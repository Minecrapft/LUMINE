"""
app/ml/baseline.py — CIE ΔE00 Nearest-Reference Baseline Predictor for LUMINE.

Given a measured pad colour (RGB) and a list of reference entries loaded from
the database (each has 'level' and 'rgb'), this module finds the closest
reference by CIEDE2000 distance and returns a structured result including
confidence derived from the margin between the best and second-best match.

This is the **default** predictor — it replaces the old RGB-Euclidean
`color_analysis.py` comparisons and the hardcoded grayscale fallback.

Public API
----------
predict_level(measured_rgb, reference_entries) → dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import math
import logging
import numpy as np

from app.ml.color import rgb_to_lab, delta_e

logger = logging.getLogger(__name__)

# Tuning constants — adjust as more real-world data becomes available
_LOW_CONFIDENCE_MARGIN_THRESHOLD = 3.0   # ΔE units; below this → ambiguous
_MIN_CONFIDENCE_DE = 80.0                # ΔE at which confidence reaches 0.0
_MAX_CONFIDENCE_DE = 0.0                 # ΔE at which confidence reaches 1.0


def _de_to_confidence(delta_e_best: float,
                       delta_e_second: Optional[float]) -> float:
    """
    Map best-match ΔE and margin to a [0, 1] confidence score.

    Logic
    -----
    - Confidence decreases as the best match is further away (absolute fit quality).
    - Confidence also decreases as the margin between best and second-best narrows
      (discriminability quality).

    Both components are combined with a geometric mean.
    """
    # Component 1: absolute fit quality (0 = perfect match, 1 = very far away)
    absolute = max(0.0, 1.0 - (delta_e_best / _MIN_CONFIDENCE_DE))

    # Component 2: margin quality
    if delta_e_second is None or not math.isfinite(delta_e_second):
        margin_conf = 1.0  # Only one reference; assume it is correct
    else:
        margin = delta_e_second - delta_e_best
        # Sigmoid-like: margin = 10 → 1.0, margin = 0 → 0.0
        margin_conf = min(1.0, margin / 10.0)

    return round(math.sqrt(absolute * margin_conf), 4)


def _delta_e_to_similarity_percent(delta_e_value: float) -> int:
    """Convert ΔE00 to a bounded, app-defined color-similarity score."""
    return round(max(0.0, 100.0 - min(delta_e_value, 100.0)))


def predict_level(
    measured_rgb: Union[Sequence, np.ndarray],
    reference_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Predict the concentration level of a strip pad from its colour.

    Parameters
    ----------
    measured_rgb : array-like of length 3
        The measured pad colour as [R, G, B] (uint8 0–255 or float 0–1).
    reference_entries : list of dict
        Each dict must have at least:
          - 'level' (str): e.g. 'negative', 'trace', '+1', '+2', ...
          - 'rgb'   (list[int]): reference [R, G, B] in 0–255.
        May also contain optional 'lab' (pre-computed Lab) for speed.

    Returns
    -------
    dict with keys:
        level            (str)           — predicted level
        delta_e_best     (float)         — ΔE00 to best reference
        delta_e_second   (float | None)  — ΔE00 to 2nd-best reference (None if only 1 ref)
        confidence       (float)         — [0, 1], higher is more reliable
        low_confidence   (bool)          — True if margin < threshold

    Raises
    ------
    ValueError  if reference_entries is empty.
    """
    if not reference_entries:
        raise ValueError("reference_entries must not be empty")

    measured_lab = rgb_to_lab(measured_rgb)

    distances = []
    for ref in reference_entries:
        # Use pre-computed Lab if present, else compute from rgb
        if ref.get("lab"):
            try:
                import json
                ref_lab = np.asarray(
                    json.loads(ref["lab"]) if isinstance(ref["lab"], str) else ref["lab"],
                    dtype=float,
                )
            except Exception:
                ref_lab = rgb_to_lab(ref["rgb"])
        else:
            ref_rgb = ref.get("rgb", [128, 128, 128])
            if isinstance(ref_rgb, str):
                try:
                    import json
                    ref_rgb = json.loads(ref_rgb)
                except Exception:
                    ref_rgb = [128, 128, 128]
            ref_lab = rgb_to_lab(ref_rgb)

        de = delta_e(measured_lab, ref_lab)
        distances.append((de, ref.get("level", "unknown")))

    distances.sort(key=lambda x: x[0])

    best_de, best_level = distances[0]

    if len(distances) > 1:
        second_de, _ = distances[1]
    else:
        second_de = None

    margin = (second_de - best_de) if second_de is not None else float("inf")
    low_confidence = margin < _LOW_CONFIDENCE_MARGIN_THRESHOLD

    confidence = _de_to_confidence(best_de, second_de)

    return {
        "level": best_level,
        "delta_e_best": round(best_de, 3),
        "delta_e_second": round(second_de, 3) if second_de is not None else None,
        "color_similarity_percent": _delta_e_to_similarity_percent(best_de),
        "confidence": confidence,
        "low_confidence": low_confidence,
        "all_distances": [
            {"level": lvl, "delta_e": round(de, 3)} for de, lvl in distances
        ],
    }
