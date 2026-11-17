"""
LUMINE Image Quality Gates

Assesses input photo and detected strip for:
- Blur (variance of Laplacian)
- Exposure (underexposure / overexposure / clipped pixels)
- Glare (saturated specular highlights)
- Resolution (minimum strip pixel size)

Produces warnings or retake requirements adhering to the Section 5 response contract.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityGateChecker:
    """
    Evaluates image and strip quality to ensure diagnostic suitability.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Thresholds
        self.min_blur_retake = float(self.config.get('min_blur_retake', 35.0))
        self.min_blur_warning = float(self.config.get('min_blur_warning', 80.0))

        self.min_brightness_retake = float(self.config.get('min_brightness_retake', 35.0))
        self.max_brightness_retake = float(self.config.get('max_brightness_retake', 230.0))
        self.max_dark_clipped_fraction = float(self.config.get('max_dark_clipped_fraction', 0.40))
        self.max_bright_clipped_fraction = float(self.config.get('max_bright_clipped_fraction', 0.20))

        self.max_glare_ratio_retake = float(self.config.get('max_glare_ratio_retake', 0.15))
        self.max_glare_ratio_warning = float(self.config.get('max_glare_ratio_warning', 0.05))

        self.min_strip_length = int(self.config.get('min_strip_length', 100))
        self.min_strip_width = int(self.config.get('min_strip_width', 12))

    def evaluate_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assesses global image quality prior to or alongside strip detection.
        Returns quality dictionary matching the API contract:
        {
            'pass': bool,
            'retake_required': bool,
            'reason': Optional[str],
            'blur': float,
            'exposure': 'ok' | 'underexposed' | 'overexposed',
            'glare': bool,
            'warnings': List[str],
            'metrics': Dict[str, Any]
        }
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                'pass': False,
                'retake_required': True,
                'reason': 'Image is empty or invalid.',
                'blur': 0.0,
                'exposure': 'underexposed',
                'glare': False,
                'warnings': ['Empty image data'],
                'metrics': {}
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        total_pixels = float(h * w)

        warnings: List[str] = []
        retake_reasons: List[str] = []

        # -------------------------------------------------------------
        # 1. Blur Check (Variance of Laplacian)
        # -------------------------------------------------------------
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.min_blur_retake:
            retake_reasons.append(f"Image is too blurry (focus score: {blur_score:.1f} < {self.min_blur_retake:.1f}).")
        elif blur_score < self.min_blur_warning:
            warnings.append(f"Image has slight blur (focus score: {blur_score:.1f}).")

        # -------------------------------------------------------------
        # 2. Exposure Check (Mean brightness & clipped pixel fractions)
        # -------------------------------------------------------------
        mean_brightness = float(np.mean(gray))
        dark_clipped = float(np.sum(gray <= 15)) / total_pixels
        bright_clipped = float(np.sum(gray >= 250)) / total_pixels

        exposure_status = 'ok'
        if mean_brightness < self.min_brightness_retake or dark_clipped > self.max_dark_clipped_fraction:
            exposure_status = 'underexposed'
            retake_reasons.append("Image is severely underexposed / too dark.")
        elif mean_brightness > self.max_brightness_retake or bright_clipped > self.max_bright_clipped_fraction:
            exposure_status = 'overexposed'
            retake_reasons.append("Image is severely overexposed / washed out.")
        elif mean_brightness < 60.0:
            warnings.append("Low lighting detected; results may have reduced contrast.")
        elif mean_brightness > 200.0:
            warnings.append("High ambient brightness detected.")

        # -------------------------------------------------------------
        # 3. Glare Check (Saturated specular highlights)
        # -------------------------------------------------------------
        glare_ratio = bright_clipped
        has_glare = False
        if glare_ratio > self.max_glare_ratio_retake:
            has_glare = True
            retake_reasons.append(f"Excessive specular glare / reflection detected ({glare_ratio * 100:.1f}%).")
        elif glare_ratio > self.max_glare_ratio_warning:
            has_glare = True
            warnings.append(f"Mild specular glare detected ({glare_ratio * 100:.1f}%).")

        retake_required = len(retake_reasons) > 0
        overall_pass = not retake_required and len(warnings) == 0

        reason_str = " ".join(retake_reasons) if retake_reasons else None

        return {
            'pass': overall_pass,
            'retake_required': retake_required,
            'reason': reason_str,
            'blur': blur_score,
            'exposure': exposure_status,
            'glare': has_glare,
            'warnings': warnings,
            'metrics': {
                'mean_brightness': mean_brightness,
                'dark_clipped_fraction': dark_clipped,
                'bright_clipped_fraction': bright_clipped,
                'resolution': (w, h)
            }
        }

    def evaluate_strip_resolution(self, strip_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates that the detected strip is sufficiently large in the image
        to permit reliable pad segmentation.
        """
        if not strip_result or not strip_result.get('detected', False):
            return False, "Strip was not detected."

        bbox = strip_result.get('bbox', (0, 0, 0, 0))
        _, _, bw, bh = bbox
        major_dim = max(bw, bh)
        minor_dim = min(bw, bh)

        if major_dim < self.min_strip_length or minor_dim < self.min_strip_width:
            return False, (
                f"Strip resolution is too low ({major_dim}x{minor_dim}px, "
                f"minimum required: {self.min_strip_length}x{self.min_strip_width}px). "
                "Please move camera closer to the strip."
            )

        return True, None
