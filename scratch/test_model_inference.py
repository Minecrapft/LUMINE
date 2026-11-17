import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import json
from ultralytics import YOLO
from app.image_processing.yolo_detector import ONNXYOLODetector

img_path = r"C:/Users/marlon/.gemini/antigravity/brain/44d3ff38-55c8-441e-841c-a707fd3ff887/.user_uploaded/media_1793282889884_2be8d76b.jpg"
pt_path = r"models/yolov8n_strip_b5303be8/weights/best.pt"
onnx_path = r"models/yolov8n_strip_b5303be8/weights/best.onnx"

names = ['Bilirubin', 'Blood', 'Glucose', 'Ketone', 'Leukocytes', 'Nitrite', 'Protein', 'SpGravity', 'Urobilinogen', 'pH', 'strip']

# 1. Ultralytics PyTorch inference
print("=== Running PyTorch YOLOv8 (.pt) Inference ===")
model = YOLO(pt_path)
results = model.predict(source=img_path, conf=0.25, save=False)
boxes = results[0].boxes
print(f"PyTorch detections count: {len(boxes)}")
for b in boxes:
    cls_id = int(b.cls[0])
    conf = float(b.conf[0])
    xyxy = b.xyxy[0].tolist()
    name = names[cls_id] if cls_id < len(names) else str(cls_id)
    print(f"  Detected: {name:<12} | Conf: {conf:.3f} | BBox: {[round(x, 1) for x in xyxy]}")

# Save PyTorch visual prediction
pred_img = results[0].plot()
out_pt_vis = "scratch/detection_result_pt.jpg"
cv2.imwrite(out_pt_vis, pred_img)
print(f"Saved PyTorch visual result to {out_pt_vis}")

# 2. OpenCV DNN ONNX inference (Raspberry Pi 5 target)
print("\n=== Running OpenCV DNN ONNX (.onnx) Inference ===")
detector = ONNXYOLODetector(model_path=onnx_path, conf_threshold=0.25)
img_bgr = cv2.imread(img_path)
onnx_detections = detector.detect(img_bgr)
print(f"ONNX cv2.dnn detections count: {len(onnx_detections)}")
vis_onnx = img_bgr.copy()
for d in onnx_detections:
    cls_id = d['class_id']
    conf = d['confidence']
    x, y, w, h = d['bbox']
    name = names[cls_id] if cls_id < len(names) else str(cls_id)
    print(f"  Detected: {name:<12} | Conf: {conf:.3f} | BBox: [x={x}, y={y}, w={w}, h={h}]")
    cv2.rectangle(vis_onnx, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label_txt = f"{name} {conf:.2f}"
    cv2.putText(vis_onnx, label_txt, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

out_onnx_vis = "scratch/detection_result_onnx.jpg"
cv2.imwrite(out_onnx_vis, vis_onnx)
print(f"Saved ONNX visual result to {out_onnx_vis}")
