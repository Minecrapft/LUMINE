"""Joint colour matching for four-pad universal pH strips."""

from typing import Any, Dict, List, Sequence
import math

from app.ml.chart_colors import SALIVA_PH4_PAD_CHART
from app.ml.color import delta_e, rgb_to_lab


def predict_saliva_ph(pad_colors: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Match four measured pad colours against the supplied pH reference chart."""
    if len(pad_colors) != 4:
        raise ValueError("A saliva pH strip requires exactly four pad colors")

    measured = []
    for color in pad_colors:
        if len(color) != 3 or any(not 0 <= int(channel) <= 255 for channel in color):
            raise ValueError("Each pad color must contain three RGB channels from 0 to 255")
        measured.append([int(channel) for channel in color])

    measured_lab = [rgb_to_lab(color) for color in measured]
    matches = []
    for level, chart_colors in SALIVA_PH4_PAD_CHART.items():
        chart_lab = [rgb_to_lab(color) for color in chart_colors]
        for reversed_order in (False, True):
            ordered_chart = list(reversed(chart_lab)) if reversed_order else chart_lab
            distances = [
                delta_e(image_lab, chart_lab_color)
                for image_lab, chart_lab_color in zip(measured_lab, ordered_chart)
            ]
            matches.append({
                "level": level,
                "reversed": reversed_order,
                "distance": sum(distances) / len(distances),
                "pad_distances": distances,
            })

    matches.sort(key=lambda match: match["distance"])
    best = matches[0]
    second_distance = matches[1]["distance"]
    margin = max(0.0, second_distance - best["distance"])
    absolute_confidence = max(0.0, 1.0 - best["distance"] / 80.0)
    margin_confidence = min(1.0, margin / 10.0)
    confidence = round(math.sqrt(absolute_confidence * margin_confidence), 4)
    similarity = round(max(0.0, 100.0 - min(best["distance"], 100.0)))
    chart_colors = SALIVA_PH4_PAD_CHART[best["level"]]

    if best["reversed"]:
        chart_colors = list(reversed(chart_colors))

    return {
        "level": best["level"],
        "delta_e_best": round(best["distance"], 3),
        "delta_e_second": round(second_distance, 3),
        "confidence": confidence,
        "low_confidence": margin < 3.0,
        "color_similarity_percent": similarity,
        "reversed_chart_order": best["reversed"],
        "pad_comparisons": [
            {
                "pad_index": index + 1,
                "measured_rgb": measured[index],
                "chart_rgb": chart_colors[index],
                "delta_e": round(best["pad_distances"][index], 3),
            }
            for index in range(4)
        ],
    }
