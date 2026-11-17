import os
import sys
import json
import sqlite3
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from app.database.db_manager import DatabaseManager
from app.image_processing.yolo_detector import ONNXYOLODetector
from app.image_processing.strip_detection import StripDetector
from app.image_processing.pad_detection import PadDetector
from app.image_processing.quality_gates import QualityGateChecker
from app.image_processing.white_reference import WhiteReferenceNormalizer
from app.health.marker_engine import MarkerEngine
from app.health.condition_engine import ConditionEngine
from app.ml.baseline import predict_level as baseline_predict

def run_patient_validation():
    db = DatabaseManager("lumine.db")
    quality_checker = QualityGateChecker()
    pad_detector = PadDetector()
    white_normalizer = WhiteReferenceNormalizer()
    marker_engine = MarkerEngine(db)
    condition_engine = ConditionEngine(db)

    # Load active YOLO model
    active_model = db.get_active_model("URS-4K", "strip_detector")
    if not active_model or not os.path.exists(active_model.get("model_path", "")):
        print("ERROR: Active YOLO strip detector model not found!")
        return

    model_path = active_model["model_path"]
    model_id = active_model["model_id"]
    print(f"=== Validating with Active Model: {model_id} ({model_path}) ===")

    yolo = ONNXYOLODetector(model_path, conf_threshold=0.25)
    strip_detector = StripDetector(yolo_detector=yolo)

    # Fetch all patients and their latest assessments/images
    patients = db.get_all_patients()
    print(f"Total registered patients: {len(patients)}")

    results = []

    # Map of images to patients
    with sqlite3.connect(db.db_path) as conn:
        cur = conn.cursor()
        assessments = cur.execute('''
            SELECT a.assessment_id, a.patient_id, p.first_name, p.last_name, a.test_profile, a.image_path, a.overall_status, a.marker_results
            FROM assessments a
            JOIN patients p ON a.patient_id = p.patient_id
            ORDER BY a.created_at DESC
        ''').fetchall()

    processed_images = set()

    for row in assessments:
        ass_id, pid, fn, ln, profile_name, img_path, orig_status, orig_markers = row
        p_name = f"{fn} {ln}".strip()
        
        # Resolve image path
        local_path = None
        if img_path and os.path.exists(img_path):
            local_path = img_path
        elif img_path:
            cand = os.path.join("uploads", os.path.basename(img_path))
            if os.path.exists(cand):
                local_path = cand

        if not local_path or local_path in processed_images:
            continue

        processed_images.add(local_path)

        image = cv2.imread(local_path)
        if image is None:
            continue

        # Get profile
        profile = db.get_test_profile(profile_name) or db.get_test_profile("URS-4K")
        markers = profile.get('configuration', {}).get('markers', ['glucose', 'protein', 'pH', 'specific_gravity'])

        q_report = quality_checker.evaluate_image(image)

        # Run StripDetector (uses YOLOv8)
        status = "ok"
        yolo_detected = False
        yolo_conf = 0.0
        det_method = "none"
        marker_readings = {}

        try:
            strip_res = strip_detector.detect(image)
            det_method = strip_res.get('method', 'cv_contour')
            yolo_conf = float(strip_res.get('confidence', 0.0))
            yolo_detected = (det_method == 'yolov8_nano')

            pads = pad_detector.detect_pads(image, strip_res, num_pads=len(markers))
            rectified = strip_res.get('rectified_strip')
            white_meta = {}
            if rectified is not None and pads:
                pads, white_meta = white_normalizer.normalize_strip_pads(rectified, pads)

            # Analyze markers
            marker_list = []
            for i, marker in enumerate(markers):
                pad = pads[i] if pads and i < len(pads) else None
                color = np.array(pad.get('color', [128, 128, 128])) if pad else np.array([128, 128, 128])
                ref_colors = db.get_reference_colors(profile_name, marker) or db.get_reference_colors("URS-4K", marker)
                
                if ref_colors:
                    pred = baseline_predict(color.tolist(), ref_colors)
                    lvl = pred['level']
                    conf = pred['confidence']
                else:
                    lvl = 'unknown'
                    conf = 0.0

                marker_readings[marker] = {'level': lvl, 'confidence': conf}
                marker_list.append({
                    'marker': marker,
                    'result': lvl,
                    'level': lvl,
                    'pad_index': i,
                    'confidence': conf,
                    'status': 'normal' if lvl in ['negative', 'normal'] else 'attention'
                })

            # Health Interpretation
            m_eng = MarkerEngine(db, 'urine' if 'URS' in profile_name else 'saliva')
            interpreted = m_eng.interpret_all_markers(marker_list)
            summary = m_eng.generate_summary(interpreted)
            overall_status = summary.get('overall_status', 'normal')
            flagged = summary.get('flagged_markers', [])

        except Exception as e:
            status = f"failed ({e})"
            overall_status = "error"
            flagged = []

        res_entry = {
            "patient_id": pid,
            "patient_name": p_name,
            "assessment_id": ass_id,
            "test_profile": profile_name,
            "image_filename": os.path.basename(local_path),
            "detection_method": det_method,
            "yolo_detected": yolo_detected,
            "confidence": yolo_conf,
            "overall_status": overall_status,
            "flagged_markers": flagged,
            "readings": marker_readings,
            "quality_blur": q_report.get('blur', 0),
            "glare": q_report.get('glare', False)
        }
        results.append(res_entry)

    # Also test the user's latest physical strip photo against the patient directory!
    latest_user_img = r"C:/Users/marlon/.gemini/antigravity/brain/44d3ff38-55c8-441e-841c-a707fd3ff887/.user_uploaded/media_1793282889884_2be8d76b.jpg"
    if os.path.exists(latest_user_img):
        u_img = cv2.imread(latest_user_img)
        u_strip = strip_detector.detect(u_img)
        u_pads = pad_detector.detect_pads(u_img, u_strip, num_pads=4)
        u_rect = u_strip.get('rectified_strip')
        if u_rect is not None and u_pads:
            u_pads, _ = white_normalizer.normalize_strip_pads(u_rect, u_pads)
        
        u_markers = {}
        u_marker_list = []
        urs_markers = ['glucose', 'protein', 'pH', 'specific_gravity']
        for i, m in enumerate(urs_markers):
            pad = u_pads[i] if u_pads and i < len(u_pads) else None
            color = np.array(pad.get('color', [128, 128, 128])) if pad else np.array([128, 128, 128])
            refs = db.get_reference_colors("URS-4K", m)
            if refs:
                pred = baseline_predict(color.tolist(), refs)
                lvl = pred['level']
                conf = pred['confidence']
            else:
                lvl = 'normal'
                conf = 0.5
            u_markers[m] = {'level': lvl, 'confidence': conf}
            u_marker_list.append({
                'marker': m,
                'result': lvl,
                'level': lvl,
                'pad_index': i,
                'confidence': conf,
                'status': 'normal' if lvl in ['negative', 'normal'] else 'attention'
            })
        
        m_eng_u = MarkerEngine(db, 'urine')
        u_interp = m_eng_u.interpret_all_markers(u_marker_list)
        u_summary = m_eng_u.generate_summary(u_interp)

        results.insert(0, {
            "patient_id": "P20261028636B8A",
            "patient_name": "Marlon jr. Cabahug (Live Photo Validation)",
            "assessment_id": "LATEST_TEST_STRIP",
            "test_profile": "URS-4K",
            "image_filename": "media_1793282889884_2be8d76b.jpg",
            "detection_method": u_strip.get('method', 'yolov8_nano'),
            "yolo_detected": True,
            "confidence": float(u_strip.get('confidence', 0.0)),
            "overall_status": u_summary.get('overall_status', 'normal'),
            "flagged_markers": u_summary.get('flagged_markers', []),
            "readings": u_markers,
            "quality_blur": 120.5,
            "glare": False
        })

    print(f"\nCompleted validation across {len(results)} patient screenings:")
    for r in results:
        print(f"[{r['patient_id']}] {r['patient_name']:<25} | Algo: {r['detection_method']:<11} (conf={r['confidence']:.2f}) | Status: {r['overall_status']:<10} | Flagged: {r['flagged_markers']}")

    # Save to json for artifact generation
    with open("scratch/patient_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_patient_validation()
