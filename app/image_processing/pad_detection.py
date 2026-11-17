"""
LUMINE pad detection — YOLO class boxes with prior-driven localization fallback.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PadDetector:
    """
    Uses trained YOLO pad boxes when available and strip geometry priors as a
    fallback. Samples inner regions to avoid substrate bleed, flags
    glare/dispersion, and generates debug overlays.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_num_pads = int(self.config.get('num_pads', 10))
        # Inner sampling ratio (e.g. 0.55 = central 55% of the pad width & height)
        self.inner_crop_ratio = float(self.config.get('inner_crop_ratio', 0.55))
        # Flag pads with dispersion (std dev across pixels) exceeding threshold
        self.variance_threshold = float(self.config.get('variance_threshold', 26.0))

    def detect_pads(
        self,
        image: np.ndarray,
        strip_info: Union[Dict[str, Any], Tuple[int, int, int, int]],
        num_pads: Optional[int] = None,
        class_names: Optional[Union[Dict[int, str], List[str]]] = None,
        marker_names: Optional[List[str]] = None,
        detect_color_regions: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detects individual reagent pads along the test strip.
        strip_info can be either:
          - A strip_result dict from StripDetector (preferred, contains rectified_strip)
          - A bounding box tuple (x, y, w, h)
        When a model result includes named YOLO detections, marker-matched boxes
        are used instead of the evenly spaced fallback.
        """
        if image is None or image.size == 0:
            return []

        pads_count = num_pads if num_pads and num_pads > 0 else self.default_num_pads

        # Case 1: Dict with rectified strip and inverse transform matrix
        if isinstance(strip_info, dict) and 'rectified_strip' in strip_info:
            model_pads = self._detect_pads_from_yolo(
                image,
                strip_info,
                class_names or {},
                marker_names,
                pads_count,
            )
            if model_pads:
                if not marker_names:
                    return model_pads
                fallback_pads = self._detect_pads_rectified(
                    image, strip_info, pads_count, detect_color_regions
                )
                pads_by_index = {}
                for index, marker in enumerate(marker_names or []):
                    if index < len(fallback_pads):
                        fallback_pad = dict(fallback_pads[index])
                        fallback_pad['marker'] = marker
                        fallback_pad['method'] = 'cv_fallback'
                        pads_by_index[index] = fallback_pad
                for model_pad in model_pads:
                    pads_by_index[model_pad['index']] = model_pad

                ordered_pads = [
                    pads_by_index[index]
                    for index in range(pads_count)
                    if index in pads_by_index
                ]
                missing_count = pads_count - len(ordered_pads)
                if missing_count:
                    logger.warning(
                        "PadDetector: YOLO and fallback localized %s of %s configured pads",
                        len(ordered_pads),
                        pads_count,
                    )
                elif len(model_pads) < pads_count:
                    logger.info(
                        "PadDetector: Filled %s missed YOLO pad detections with geometric fallback",
                        pads_count - len(model_pads),
                    )
                return ordered_pads
            return self._detect_pads_rectified(
                image, strip_info, pads_count, detect_color_regions
            )

        # Case 2: Bounding box tuple fallback (x, y, w, h)
        if isinstance(strip_info, (tuple, list)) and len(strip_info) == 4:
            return self._detect_pads_bbox_fallback(image, tuple(strip_info), pads_count)

        logger.warning(f"PadDetector: Unrecognized strip_info format {type(strip_info)}")
        return []

    @staticmethod
    def _normalize_marker_name(name: str) -> str:
        normalized = ''.join(character for character in name.casefold() if character.isalnum())
        aliases = {
            'ketone': 'ketones',
            'spgravity': 'specificgravity',
        }
        return aliases.get(normalized, normalized)

    def _detect_pads_from_yolo(
        self,
        image: np.ndarray,
        strip_result: Dict[str, Any],
        class_names: Union[Dict[int, str], List[str]],
        marker_names: Optional[List[str]],
        expected_count: int,
    ) -> List[Dict[str, Any]]:
        detections = strip_result.get('yolo_detections') or []
        if not detections:
            return []

        strip_x, strip_y, strip_w, strip_h = strip_result.get('bbox', (0, 0, 0, 0))
        image_h, image_w = image.shape[:2]
        rectified = strip_result['rectified_strip']
        transform = strip_result.get('transform_matrix')
        h_rect, w_rect = rectified.shape[:2]
        reverse_order = False
        if transform is None:
            mid = w_rect // 2
            left_std = float(np.mean(np.std(rectified[:, :mid], axis=(0, 1))))
            right_std = float(np.mean(np.std(rectified[:, mid:], axis=(0, 1))))
            reverse_order = right_std > left_std * 1.3
        candidates = []

        for detection in detections:
            class_name = detection.get('class_name')
            class_id = detection.get('class_id')
            if class_name is None:
                if isinstance(class_names, dict):
                    class_name = class_names.get(class_id, class_names.get(str(class_id)))
                elif isinstance(class_names, (list, tuple)) and isinstance(class_id, int):
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
            is_strip_class = (
                isinstance(class_name, str)
                and self._normalize_marker_name(class_name) == 'strip'
            ) or (class_name is None and class_id == 10)
            if is_strip_class:
                continue

            bbox = detection.get('bbox', ())
            if len(bbox) != 4:
                continue
            x, y, width, height = (int(value) for value in bbox)
            if width <= 0 or height <= 0:
                continue
            center = (x + width // 2, y + height // 2)
            if not (
                strip_x <= center[0] <= strip_x + strip_w
                and                 strip_y <= center[1] <= strip_y + strip_h
            ):
                continue

            x1 = max(0, min(image_w, x))
            y1 = max(0, min(image_h, y))
            x2 = max(x1, min(image_w, x + width))
            y2 = max(y1, min(image_h, y + height))
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if transform is not None:
                point = np.array([[[center[0], center[1]]]], dtype=np.float32)
                canonical_x = float(cv2.perspectiveTransform(point, transform)[0, 0, 0])
            else:
                canonical_x = float(
                    (center[1] - strip_y) / max(strip_h, 1) * w_rect
                    if strip_h > strip_w
                    else (center[0] - strip_x) / max(strip_w, 1) * w_rect
                )
            if reverse_order:
                canonical_x = w_rect - canonical_x

            confidence = float(detection.get('confidence', 0.0))
            duplicate_index = next((
                index for index, candidate in enumerate(candidates)
                if self._boxes_overlap(candidate['bbox'], (x1, y1, x2 - x1, y2 - y1))
            ), None)
            if duplicate_index is not None:
                if candidates[duplicate_index]['confidence'] >= confidence:
                    continue
                candidates.pop(duplicate_index)

            # Sample the box's center to reduce background contamination.
            inner_x1 = int(roi.shape[1] * 0.2)
            inner_x2 = max(inner_x1 + 1, int(roi.shape[1] * 0.8))
            inner_y1 = int(roi.shape[0] * 0.2)
            inner_y2 = max(inner_y1 + 1, int(roi.shape[0] * 0.8))
            inner_roi = roi[inner_y1:inner_y2, inner_x1:inner_x2]
            mean_bgr = np.mean(inner_roi, axis=(0, 1))
            std_bgr = np.std(inner_roi, axis=(0, 1))
            avg_rgb = [
                int(np.clip(mean_bgr[2], 0, 255)),
                int(np.clip(mean_bgr[1], 0, 255)),
                int(np.clip(mean_bgr[0], 0, 255)),
            ]
            dispersion = float(np.mean(std_bgr))
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            candidates.append({
                'canonical_x': canonical_x,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'corners': corners,
                'color': avg_rgb,
                'color_normalized': [value / 255.0 for value in avg_rgb],
                'dispersion': dispersion,
                'high_variance': dispersion > self.variance_threshold,
                'method': 'yolov8_nano',
            })

        candidates.sort(key=lambda candidate: candidate['canonical_x'])
        pads = []
        for index, candidate in enumerate(candidates[:expected_count]):
            candidate['index'] = index
            if marker_names and index < len(marker_names):
                candidate['marker'] = marker_names[index]
            pads.append(candidate)
        if pads:
            logger.info(
                "PadDetector: YOLO localized %s of %s configured pads in strip order",
                len(pads),
                expected_count,
            )
        return pads

    @staticmethod
    def _boxes_overlap(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> bool:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[0] + first[2], second[0] + second[2])
        y2 = min(first[1] + first[3], second[1] + second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return False
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        union = first_area + second_area - intersection
        return union > 0 and intersection / union >= 0.5

    def _detect_pads_rectified(
        self,
        image: np.ndarray,
        strip_result: Dict[str, Any],
        num_pads: int,
        detect_color_regions: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detects pads along the canonically oriented horizontal rectified strip.
        """
        rectified = strip_result['rectified_strip']
        inv_M = strip_result.get('inv_transform_matrix')
        h_rect, w_rect = rectified.shape[:2]

        # In canonical dipstick strips, the active pad section occupies
        # approximately the first 72% to 78% of the strip length; the rest is the handle.
        start_margin = float(w_rect) * 0.02
        active_ratio = 0.75 if num_pads > 1 else 0.85
        active_width = float(w_rect) * active_ratio - start_margin
        pad_pitch = active_width / float(num_pads)

        pad_w = pad_pitch * 0.68
        pad_h = float(h_rect) * 0.62
        pad_y0 = (float(h_rect) - pad_h) / 2.0

        # Sample substrate color from the handle (rightmost 15%)
        hx1 = int(w_rect * 0.85)
        handle_slice = rectified[:, hx1:]
        sub_bgr = np.mean(handle_slice, axis=(0, 1)) if handle_slice.size > 0 else np.array([245.0, 245.0, 245.0])

        # Analyze color difference from substrate across horizontal strip centerline
        mid_y1 = int(h_rect * 0.25)
        mid_y2 = int(h_rect * 0.75)
        strip_center_slice = rectified[mid_y1:mid_y2, :, :].astype(np.float32)
        diff_from_sub = np.mean(np.linalg.norm(strip_center_slice - sub_bgr, axis=2), axis=0)
        detected_regions = (
            self._find_color_pad_regions(diff_from_sub, w_rect, num_pads)
            if detect_color_regions else None
        )
        if detect_color_regions and detected_regions is None:
            logger.warning(
                "PadDetector: Could not identify %s distinct color regions on the strip",
                num_pads,
            )
            return []

        pads = []
        for i in range(num_pads):
            if detected_regions is not None:
                rx1, rx2 = detected_regions[i]
                cx = (rx1 + rx2) / 2.0
            else:
                # Prior center of pad i along strip length
                prior_cx = start_margin + (i + 0.5) * pad_pitch

                # Refine center using local center-of-mass of color difference
                search_radius = int(pad_pitch * 0.25)
                s_min = max(0, int(prior_cx - search_radius))
                s_max = min(w_rect - 1, int(prior_cx + search_radius))

                if s_max > s_min:
                    window = diff_from_sub[s_min:s_max+1]
                    w_min = np.min(window)
                    w_weights = np.maximum(window - w_min, 0.0)
                    sum_w = np.sum(w_weights)
                    if sum_w > 1e-3:
                        indices = np.arange(len(window), dtype=np.float32)
                        offset = (np.sum(indices * w_weights) / sum_w) - (len(window) / 2.0)
                        cx = prior_cx + np.clip(offset * 0.35, -pad_pitch * 0.12, pad_pitch * 0.12)
                    else:
                        cx = prior_cx
                else:
                    cx = prior_cx

            cy = pad_y0 + pad_h / 2.0

            if detected_regions is None:
                rx1 = max(0.0, cx - pad_w / 2.0)
                rx2 = min(float(w_rect - 1), cx + pad_w / 2.0)
            ry1 = max(0.0, cy - pad_h / 2.0)
            ry2 = min(float(h_rect - 1), cy + pad_h / 2.0)

            # Sample from INNER region (central 50%) to strictly avoid edge bleed & shadows
            inner_w = (rx2 - rx1) * 0.50
            inner_h = (ry2 - ry1) * 0.50
            ix1 = int(cx - inner_w / 2.0)
            ix2 = int(cx + inner_w / 2.0)
            iy1 = int(cy - inner_h / 2.0)
            iy2 = int(cy + inner_h / 2.0)

            ix1 = max(0, min(w_rect - 1, ix1))
            ix2 = max(ix1 + 1, min(w_rect, ix2))
            iy1 = max(0, min(h_rect - 1, iy1))
            iy2 = max(iy1 + 1, min(h_rect, iy2))

            inner_roi = rectified[iy1:iy2, ix1:ix2]
            if inner_roi.size == 0:
                inner_roi = rectified[int(ry1):int(ry2), int(rx1):int(rx2)]

            # Average BGR -> RGB
            if inner_roi.size > 0:
                mean_bgr = np.mean(inner_roi, axis=(0, 1))
                std_bgr = np.std(inner_roi, axis=(0, 1))
                avg_rgb = [int(np.clip(mean_bgr[2], 0, 255)),
                           int(np.clip(mean_bgr[1], 0, 255)),
                           int(np.clip(mean_bgr[0], 0, 255))]
                dispersion = float(np.mean(std_bgr))
            else:
                avg_rgb = [128, 128, 128]
                dispersion = 0.0

            high_variance = bool(dispersion > self.variance_threshold)

            # Map pad rectangle corners back to original image space
            rect_corners = np.array([
                [[rx1, ry1]],
                [[rx2, ry1]],
                [[rx2, ry2]],
                [[rx1, ry2]]
            ], dtype=np.float32)

            if inv_M is not None:
                orig_corners = cv2.perspectiveTransform(rect_corners, inv_M)
                orig_pts = orig_corners.reshape(-1, 2)
                min_ox = float(np.min(orig_pts[:, 0]))
                min_oy = float(np.min(orig_pts[:, 1]))
                max_ox = float(np.max(orig_pts[:, 0]))
                max_oy = float(np.max(orig_pts[:, 1]))
                orig_bbox = (int(min_ox), int(min_oy), int(max_ox - min_ox), int(max_oy - min_oy))
                orig_center = (int(np.mean(orig_pts[:, 0])), int(np.mean(orig_pts[:, 1])))
                corners_list = orig_pts.tolist()
            else:
                orig_bbox = (int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1))
                orig_center = (int(cx), int(cy))
                corners_list = rect_corners.reshape(-1, 2).tolist()

            pads.append({
                'index': i,
                'bbox': orig_bbox,
                'center': orig_center,
                'corners': corners_list,
                'rectified_bbox': (int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1)),
                'rectified_center': (float(cx), float(cy)),
                'color': avg_rgb,
                'color_normalized': [c / 255.0 for c in avg_rgb],
                'dispersion': dispersion,
                'high_variance': high_variance,
                'confidence': 0.95 if not high_variance else 0.70
            })

        logger.info(f"PadDetector: Found {len(pads)} pads along rectified strip")
        return pads

    @staticmethod
    def _find_color_pad_regions(
        color_difference: np.ndarray,
        strip_length: int,
        expected_count: int,
    ) -> Optional[List[Tuple[float, float]]]:
        """Find colored pad runs along the rectified strip's long axis."""
        search_end = int(strip_length * 0.78)
        profile = color_difference[:search_end].astype(np.float32)
        if profile.size == 0:
            return None

        smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (1, 7), 0).ravel()
        baseline = float(np.median(smoothed[int(strip_length * 0.72):]))
        active_max = float(np.max(smoothed))
        threshold = max(baseline + 8.0, baseline + (active_max - baseline) * 0.16)
        active = smoothed > threshold

        segments = []
        start = None
        for index, is_active in enumerate(active):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                segments.append((start, index))
                start = None
        if start is not None:
            segments.append((start, len(active)))

        merged = []
        for segment in segments:
            if merged and segment[0] - merged[-1][1] <= 6:
                merged[-1] = (merged[-1][0], segment[1])
            else:
                merged.append(segment)

        min_width = max(6, int(strip_length * 0.015))
        max_width = int(strip_length * 0.18)
        candidates = [
            segment for segment in merged
            if min_width <= segment[1] - segment[0] <= max_width
        ]
        if len(candidates) < expected_count:
            return None

        strongest = sorted(
            candidates,
            key=lambda segment: float(
                np.sum(smoothed[segment[0]:segment[1]] - baseline)
            ),
            reverse=True,
        )[:expected_count]
        strongest.sort()
        return [(float(start), float(end)) for start, end in strongest]

    def _detect_pads_bbox_fallback(
        self,
        image: np.ndarray,
        strip_bbox: Tuple[int, int, int, int],
        num_pads: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback when only a bounding box is supplied.
        """
        x, y, w, h = strip_bbox
        h_img, w_img = image.shape[:2]

        # Ensure bbox fits within image
        x = max(0, min(w_img - 1, x))
        y = max(0, min(h_img - 1, y))
        w = max(10, min(w_img - x, w))
        h = max(10, min(h_img - y, h))

        strip_roi = image[y:y+h, x:x+w]
        is_horizontal = w >= h

        pads = []
        if is_horizontal:
            active_w = w * 0.75
            pad_pitch = active_w / float(num_pads)
            pad_w = int(pad_pitch * 0.75)
            pad_h = int(h * 0.70)
            py = y + int((h - pad_h) / 2)

            for i in range(num_pads):
                px = x + int(i * pad_pitch + (pad_pitch - pad_w) / 2)
                roi = image[py:py+pad_h, px:px+pad_w]
                if roi.size > 0:
                    mean_bgr = np.mean(roi, axis=(0, 1))
                    avg_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
                    disp = float(np.mean(np.std(roi, axis=(0, 1))))
                else:
                    avg_rgb = [128, 128, 128]
                    disp = 0.0

                pads.append({
                    'index': i,
                    'bbox': (px, py, pad_w, pad_h),
                    'center': (px + pad_w // 2, py + pad_h // 2),
                    'corners': [[px, py], [px + pad_w, py], [px + pad_w, py + pad_h], [px, py + pad_h]],
                    'color': avg_rgb,
                    'color_normalized': [c / 255.0 for c in avg_rgb],
                    'dispersion': disp,
                    'high_variance': disp > self.variance_threshold,
                    'confidence': 0.85
                })
        else:
            active_h = h * 0.75
            pad_pitch = active_h / float(num_pads)
            pad_h = int(pad_pitch * 0.75)
            pad_w = int(w * 0.70)
            px = x + int((w - pad_w) / 2)

            for i in range(num_pads):
                py = y + int(i * pad_pitch + (pad_pitch - pad_h) / 2)
                roi = image[py:py+pad_h, px:px+pad_w]
                if roi.size > 0:
                    mean_bgr = np.mean(roi, axis=(0, 1))
                    avg_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
                    disp = float(np.mean(np.std(roi, axis=(0, 1))))
                else:
                    avg_rgb = [128, 128, 128]
                    disp = 0.0

                pads.append({
                    'index': i,
                    'bbox': (px, py, pad_w, pad_h),
                    'center': (px + pad_w // 2, py + pad_h // 2),
                    'corners': [[px, py], [px + pad_w, py], [px + pad_w, py + pad_h], [px, py + pad_h]],
                    'color': avg_rgb,
                    'color_normalized': [c / 255.0 for c in avg_rgb],
                    'dispersion': disp,
                    'high_variance': disp > self.variance_threshold,
                    'confidence': 0.85
                })

        return pads

    def extract_pad_colors(self, image: np.ndarray, pads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensures colors are extracted for all pads. Maintained for backward compatibility.
        """
        for pad in pads:
            if 'color' not in pad or pad['color'] is None:
                x, y, w, h = pad['bbox']
                roi = image[y:y+h, x:x+w]
                if roi.size > 0:
                    mean_bgr = np.mean(roi, axis=(0, 1))
                    avg_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
                else:
                    avg_rgb = [128, 128, 128]
                pad['color'] = avg_rgb
                pad['color_normalized'] = [c / 255.0 for c in avg_rgb]
        return pads

    def generate_debug_overlay(
        self,
        image: np.ndarray,
        strip_result: Dict[str, Any],
        pads: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generates an annotated debug overlay image showing:
          - Detected strip boundary quad (green outline)
          - Individual pad bounding boxes (cyan outline)
          - Pad indices (#0, #1, ...) and center crosshairs
          - Detection confidence banner
        """
        overlay = image.copy()

        # 1. Draw strip boundary polygon
        if 'corners' in strip_result and strip_result['corners']:
            corners = np.array(strip_result['corners'], dtype=np.int32)
            cv2.polylines(overlay, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # 2. Draw each pad
        for pad in pads:
            pad_idx = pad.get('index', 0)
            center = pad.get('center', (0, 0))
            is_high_var = pad.get('high_variance', False)

            # Box color: cyan normally, orange/yellow if high variance
            box_color = (0, 165, 255) if is_high_var else (255, 255, 0)

            # If polygon corners available, draw quadrilateral
            if 'corners' in pad and len(pad['corners']) == 4:
                pts = np.array(pad['corners'], dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=box_color, thickness=2)
            else:
                x, y, w, h = pad['bbox']
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)

            # Center crosshair / dot
            cx, cy = center
            cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            # Label with index
            label = f"#{pad_idx}"
            cv2.putText(
                overlay, label, (int(cx) - 10, max(15, int(cy) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 3. Confidence banner
        conf = strip_result.get('confidence', 1.0)
        banner_text = f"Strip Detection: {conf * 100:.1f}% confidence ({strip_result.get('method', 'auto')})"
        cv2.putText(
            overlay, banner_text, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        return overlay
