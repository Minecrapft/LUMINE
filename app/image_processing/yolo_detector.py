"""
LUMINE — ONNX YOLOv8 strip and pad detector.
Uses cv2.dnn for inference without PyTorch or Ultralytics at runtime.
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ONNXYOLODetector:
    """
    Lightweight YOLOv8 detector using OpenCV DNN.
    Designed for Raspberry Pi 5 (<60ms latency, ~0 extra RAM overhead).
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        class_names: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.class_names = class_names or {}
        self.net = None
        self._load_network()

    def _load_network(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        
        logger.info(f"Loading YOLOv8 ONNX model via cv2.dnn: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Use CPU backend optimal for ARM/x86
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on image.
        Returns list of detected bounding boxes:
        [{
            'bbox': [x, y, w, h],
            'confidence': float,
            'class_id': int,
            'class_name': str
        }]
        """
        if self.net is None:
            return []

        h_orig, w_orig = image.shape[:2]

        # Match YOLO's letterbox preprocessing so inference coordinates remain
        # consistent with the training images' aspect ratios.
        scale = min(self.input_size / w_orig, self.input_size / h_orig)
        resized_w = int(round(w_orig * scale))
        resized_h = int(round(h_orig * scale))
        pad_x = (self.input_size - resized_w) // 2
        pad_y = (self.input_size - resized_h) // 2
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full(
            (self.input_size, self.input_size, image.shape[2]),
            114,
            dtype=image.dtype,
        )
        letterboxed[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()  # Shape: (1, 4 + num_classes, 8400) or similar

        # Parse YOLOv8 output
        output = outputs[0]
        if output.shape[0] < output.shape[1]:
            output = output.transpose()  # Transpose to (8400, 4 + num_classes)

        boxes = []
        confidences = []
        class_ids = []

        for row in output:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf >= self.conf_threshold:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int(round((cx - 0.5 * w - pad_x) / scale))
                top = int(round((cy - 0.5 * h - pad_y) / scale))
                right = int(round((cx + 0.5 * w - pad_x) / scale))
                bottom = int(round((cy + 0.5 * h - pad_y) / scale))

                # Clamp to image boundaries
                left = max(0, min(left, w_orig - 1))
                top = max(0, min(top, h_orig - 1))
                right = max(left + 1, min(right, w_orig))
                bottom = max(top + 1, min(bottom, h_orig))
                width = right - left
                height = bottom - top

                boxes.append([left, top, width, height])
                confidences.append(conf)
                class_ids.append(class_id)

        if not boxes:
            return []

        results = []
        for class_id in set(class_ids):
            class_indices = [i for i, detected_class in enumerate(class_ids) if detected_class == class_id]
            class_boxes = [boxes[i] for i in class_indices]
            class_confidences = [confidences[i] for i in class_indices]
            indices = cv2.dnn.NMSBoxes(
                class_boxes, class_confidences, self.conf_threshold, self.nms_threshold
            )
            for class_index in indices.flatten() if len(indices) > 0 else []:
                idx = class_indices[int(class_index)]
                class_name = self._class_name(class_id)
                detection = {
                    'bbox': boxes[idx],
                    'confidence': float(confidences[idx]),
                    'class_id': int(class_id),
                }
                if class_name is not None:
                    detection['class_name'] = class_name
                results.append(detection)

        return results

    def _class_name(self, class_id: int) -> Optional[str]:
        if isinstance(self.class_names, dict):
            name = self.class_names.get(class_id, self.class_names.get(str(class_id)))
        elif isinstance(self.class_names, (list, tuple)) and class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = None
        return str(name) if name is not None else None
