"""
LUMINE Strip Detection — Computer Vision Strip Localization & Rectification
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StripNotFound(Exception):
    """Raised when no test strip is detected in the image with sufficient confidence."""
    pass


# Backward-compatible alias
StripNotFoundError = StripNotFound


def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as: [top-left, top-right, bottom-right, bottom-left].
    pts shape: (4, 2)
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


class StripDetector:
    """
    Robust contour-based strip detector with perspective rectification and
    confidence scoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, yolo_detector: Optional[Any] = None):
        self.config = config or {}
        self.yolo_detector = yolo_detector
        # Geometrical priors for medical dipstick strips
        self.min_aspect_ratio = float(self.config.get('min_aspect_ratio', 2.0))
        self.max_aspect_ratio = float(self.config.get('max_aspect_ratio', 35.0))
        self.min_area_ratio = float(self.config.get('min_area_ratio', 0.001))
        self.max_area_ratio = float(self.config.get('max_area_ratio', 0.85))
        self.confidence_threshold = float(self.config.get('confidence_threshold', 0.35))
        # Standard rectified strip dimensions (canonical horizontal: length x width)
        self.target_length = int(self.config.get('target_length', 480))
        self.target_width = int(self.config.get('target_width', 60))

    def detect(self, image: np.ndarray, manual_corners: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Detect strip in image.
        If manual_corners (4 points) provided, rectifies using those corners directly.
        Raises StripNotFound if auto-detection fails to find a plausible strip.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise StripNotFound("Input image is empty or invalid")

        h_img, w_img = image.shape[:2]
        img_area = float(h_img * w_img)

        # ------------------------------------------------------------------
        # 1. Manual Corners Path (Fallback / UI Drag-and-Drop)
        # ------------------------------------------------------------------
        if manual_corners is not None and len(manual_corners) == 4:
            pts = np.array(manual_corners, dtype=np.float32)
            ordered_pts = order_quad_corners(pts)
            rectified, M, M_inv = self._rectify_strip(image, ordered_pts)
            rectified, M, M_inv = self._orient_strip_canonically(
                rectified, transform_matrix=M, inverse_transform_matrix=M_inv
            )

            min_x = float(np.min(ordered_pts[:, 0]))
            min_y = float(np.min(ordered_pts[:, 1]))
            max_x = float(np.max(ordered_pts[:, 0]))
            max_y = float(np.max(ordered_pts[:, 1]))
            bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

            return {
                'detected': True,
                'confidence': 1.0,
                'bbox': bbox,
                'corners': ordered_pts.tolist(),
                'rectified_strip': rectified,
                'transform_matrix': M,
                'inv_transform_matrix': M_inv,
                'orientation': 'horizontal',
                'method': 'manual',
                'aspect_ratio': float((max_x - min_x) / max(max_y - min_y, 1.0))
            }

        # ------------------------------------------------------------------
        # 2. YOLOv8 Deep Learning Detector Path (Milestone 6: Raspberry Pi 5)
        # ------------------------------------------------------------------
        if self.yolo_detector is not None:
            try:
                yolo_dets = self.yolo_detector.detect(image)
                named_strip_dets = [
                    detection for detection in yolo_dets
                    if str(detection.get('class_name', '')).strip().casefold() == 'strip'
                ]
                has_class_names = any(detection.get('class_name') for detection in yolo_dets)
                strip_dets = named_strip_dets
                if not strip_dets and not has_class_names:
                    strip_dets = [
                        detection for detection in yolo_dets
                        if detection.get('class_id') == 10
                    ]
                    if not strip_dets and yolo_dets:
                        strip_dets = sorted(
                            yolo_dets,
                            key=lambda detection: detection.get('confidence', 0),
                            reverse=True,
                        )

                if strip_dets:
                    best = strip_dets[0]
                    if best.get('confidence', 0) >= self.confidence_threshold:
                        x, y, w, h = best['bbox']
                        # Construct corners
                        c_pts = np.array([
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ], dtype=np.float32)
                        ordered_c = order_quad_corners(c_pts)
                        rectified, M, M_inv = self._rectify_strip(image, ordered_c)
                        rectified, M, M_inv = self._orient_strip_canonically(
                            rectified,
                            yolo_detections=yolo_dets,
                            transform_matrix=M,
                            inverse_transform_matrix=M_inv,
                        )

                        logger.info(f"StripDetector: YOLO detection success (conf={best['confidence']:.3f})")
                        return {
                            'detected': True,
                            'confidence': float(best['confidence']),
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'corners': ordered_c.tolist(),
                            'rectified_strip': rectified,
                            'transform_matrix': M,
                            'inv_transform_matrix': M_inv,
                            'orientation': 'horizontal',
                            'method': 'yolov8_nano',
                            'yolo_detections': yolo_dets,
                            'aspect_ratio': float(max(w, h) / max(min(w, h), 1.0))
                        }
            except Exception as yolo_err:
                logger.warning(f"StripDetector: YOLO inference failed, using CV contour fallback: {yolo_err}")

        # ------------------------------------------------------------------
        # 2. Multi-cue Candidate Segmentation
        # ------------------------------------------------------------------
        candidates = self._find_candidate_strips(image, img_area)

        if not candidates:
            logger.warning("StripDetector: No candidate contours found")
            raise StripNotFound("No test strip contour detected in image")

        # Pick candidate with highest confidence score
        best_candidate = max(candidates, key=lambda c: c['confidence'])
        logger.info(f"StripDetector: Best candidate confidence = {best_candidate['confidence']:.3f}")

        if best_candidate['confidence'] < self.confidence_threshold:
            raise StripNotFound(
                f"Detection confidence ({best_candidate['confidence']:.2f}) below threshold ({self.confidence_threshold:.2f})"
            )

        ordered_corners = best_candidate['ordered_corners']
        rectified, M, M_inv = self._rectify_strip(image, ordered_corners)
        rectified, M, M_inv = self._orient_strip_canonically(
            rectified, transform_matrix=M, inverse_transform_matrix=M_inv
        )

        return {
            'detected': True,
            'confidence': float(best_candidate['confidence']),
            'bbox': best_candidate['bbox'],
            'corners': ordered_corners.tolist(),
            'rectified_strip': rectified,
            'transform_matrix': M,
            'inv_transform_matrix': M_inv,
            'orientation': 'horizontal',
            'method': 'cv_contour',
            'aspect_ratio': float(best_candidate['aspect_ratio'])
        }

    def _find_candidate_strips(self, image: np.ndarray, img_area: float) -> List[Dict[str, Any]]:
        """
        Runs multi-pass segmentation (edges, adaptive threshold, saturation)
        to discover plausible rectangular dipstick contours.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Cue 1: Morphological gradient + Canny edges with closure
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_grad = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel3)
        edges = cv2.Canny(morph_grad, 25, 90)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel5)

        # Cue 2: Adaptive thresholding (both standard and inverted)
        thresh_adapt_inv = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
        )

        # Cue 3: Color Saturation map (for white strips on light or colored backgrounds)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            sat_blurred = cv2.GaussianBlur(sat, (5, 5), 0)
            _, sat_thresh = cv2.threshold(sat_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            sat_thresh = np.zeros_like(gray)

        # Cue 4: Otsu thresholding
        _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_maps = [
            edges_closed,
            thresh_adapt_inv,
            thresh_adapt,
            otsu_inv,
            otsu_norm,
            sat_thresh
        ]

        seen_centers = []
        candidates = []

        for bmap in binary_maps:
            contours, _ = cv2.findContours(bmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                area_ratio = area / max(img_area, 1.0)
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                if w <= 0 or h <= 0:
                    continue

                major = max(w, h)
                minor = min(w, h)
                aspect = major / max(minor, 1.0)

                if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                    continue

                # Rectangularity: contour area / minAreaRect bounding area
                rect_area = max(w * h, 1.0)
                rectangularity = min(area / rect_area, 1.0)
                if rectangularity < 0.40:
                    continue

                # Deduplicate candidates with very close centers
                is_duplicate = False
                for prev_cx, prev_cy in seen_centers:
                    if (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2 < (minor * 0.5) ** 2:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                seen_centers.append((cx, cy))

                # Box corners
                box = cv2.boxPoints(rect)
                ordered_box = order_quad_corners(box)

                # Confidence calculation based on geometry
                # 1. Aspect ratio fit (ideal strip aspect ratio ~ 5.0 to 18.0)
                if 5.0 <= aspect <= 18.0:
                    ar_score = 1.0
                elif 3.0 <= aspect <= 25.0:
                    ar_score = 0.8
                else:
                    ar_score = 0.5

                # 2. Rectangularity score
                rect_score = min(max((rectangularity - 0.4) / 0.5, 0.0), 1.0)

                # 3. Size score (strip typically 0.5% - 35% of image)
                if 0.005 <= area_ratio <= 0.35:
                    size_score = 1.0
                else:
                    size_score = 0.6

                confidence = 0.40 * ar_score + 0.35 * rect_score + 0.25 * size_score

                # Bounding box in original image
                x, y, bw, bh = cv2.boundingRect(cnt)

                candidates.append({
                    'confidence': confidence,
                    'contour': cnt,
                    'bbox': (x, y, bw, bh),
                    'ordered_corners': ordered_box,
                    'aspect_ratio': aspect,
                    'area': area
                })

        return candidates

    def _rectify_strip(self, image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Warps the quadrilateral defined by 4 corners into an axis-aligned horizontal
        strip of size (target_length x target_width).
        Returns: (rectified_strip, transform_matrix, inv_transform_matrix)
        """
        corners = np.asarray(corners, dtype=np.float32)

        # Target rectangle: horizontal orientation [0,0] to [target_length, target_width]
        dst_pts = np.array([
            [0, 0],
            [self.target_length, 0],
            [self.target_length, self.target_width],
            [0, self.target_width]
        ], dtype=np.float32)

        # Check if the input corners are oriented vertically (height > width)
        # If so, map corners such that the major axis aligns with target_length
        edge1 = np.linalg.norm(corners[1] - corners[0])
        edge2 = np.linalg.norm(corners[3] - corners[0])

        if edge2 > edge1:
            # Major axis is along edge2 -> reorder corners [tl, bl, br, tr] to align horizontally
            reordered_corners = np.array([corners[0], corners[3], corners[2], corners[1]], dtype=np.float32)
        else:
            reordered_corners = corners

        M = cv2.getPerspectiveTransform(reordered_corners, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, reordered_corners)

        rectified = cv2.warpPerspective(
            image, M, (self.target_length, self.target_width),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rectified, M, M_inv

    def _orient_strip_canonically(
        self,
        rectified: np.ndarray,
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
        transform_matrix: Optional[np.ndarray] = None,
        inverse_transform_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Standardizes horizontal strip orientation so that pads are on the LEFT
        and the blank handle is on the RIGHT. Uses YOLO pad locations when
        available; image variance is only a fallback orientation cue.
        """
        w = rectified.shape[1]
        h = rectified.shape[0]
        pad_centers_x = []
        if transform_matrix is not None:
            for detection in yolo_detections or []:
                class_name = str(detection.get('class_name') or '').strip().casefold()
                if class_name == 'strip':
                    continue
                bbox = detection.get('bbox', ())
                if len(bbox) != 4:
                    continue
                x, y, box_w, box_h = (float(value) for value in bbox)
                point = np.array(
                    [[[x + box_w / 2.0, y + box_h / 2.0]]],
                    dtype=np.float32,
                )
                canonical_point = cv2.perspectiveTransform(
                    point, transform_matrix
                )[0, 0]
                canonical_x = float(canonical_point[0])
                canonical_y = float(canonical_point[1])
                if 0 <= canonical_x < w and 0 <= canonical_y < h:
                    pad_centers_x.append(canonical_x)

        if len(pad_centers_x) >= 3:
            reverse = float(np.median(pad_centers_x)) >= w / 2.0
        else:
            mid = w // 2
            left_std = float(np.mean(np.std(rectified[:, :mid], axis=(0, 1))))
            right_std = float(np.mean(np.std(rectified[:, mid:], axis=(0, 1))))
            reverse = right_std > left_std * 1.3

        if not reverse:
            return rectified, transform_matrix, inverse_transform_matrix

        rotation = np.array(
            [
                [-1.0, 0.0, w - 1.0],
                [0.0, -1.0, h - 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotation_inverse = np.linalg.inv(rotation)
        updated_transform = (
            rotation @ transform_matrix
            if transform_matrix is not None else None
        )
        updated_inverse = (
            inverse_transform_matrix @ rotation_inverse
            if inverse_transform_matrix is not None else None
        )
        return (
            cv2.rotate(rectified, cv2.ROTATE_180),
            updated_transform,
            updated_inverse,
        )
