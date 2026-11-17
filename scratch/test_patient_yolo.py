import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from app.image_processing.yolo_detector import ONNXYOLODetector
from app.image_processing.strip_detection import StripDetector

model_path = "models/yolov8n_strip_b5303be8/weights/best.onnx"
img_path = "uploads/capture_20261028_102237_26ea59ea.jpg"

if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    sys.exit(1)

img = cv2.imread(img_path)
print(f"Loaded image: {img_path}, shape={img.shape}")

# Test raw ONNX detector
onnx = ONNXYOLODetector(model_path, conf_threshold=0.25)
dets = onnx.detect(img)
print(f"Raw ONNX detections: {len(dets)}")
for d in dets:
    print(" ", d)

# Test integrated StripDetector with YOLO
sd = StripDetector(yolo_detector=onnx)
res = sd.detect(img)
print("StripDetector Result:")
print("  method:", res.get('method'))
print("  confidence:", res.get('confidence'))
print("  bbox:", res.get('bbox'))
print("  rectified shape:", res.get('rectified_strip').shape if res.get('rectified_strip') is not None else None)

