"""
LUMINE Flask Routes
"""

import json
import os
import uuid
import logging
import sqlite3
import math
import ast
import re
from datetime import datetime
from flask import (
    Blueprint, Response, current_app, jsonify, render_template, request,
    send_file, stream_with_context,
)
import cv2
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)


def _read_yolo_class_names(model_row, upload_dir):
    """Get the trained class order from model metadata or its imported dataset."""
    metrics = model_row.get('metrics') or {}
    class_names = metrics.get('class_names')
    if isinstance(class_names, dict):
        return [str(class_names[key]) for key in sorted(class_names, key=lambda key: int(key))]
    if isinstance(class_names, list) and class_names:
        return [str(name) for name in class_names]

    for dataset_id in model_row.get('dataset_ids') or []:
        data_yaml = os.path.join(upload_dir, 'yolo_datasets', dataset_id, 'data.yaml')
        try:
            with open(data_yaml, encoding='utf-8') as yaml_file:
                lines = yaml_file.readlines()
        except OSError:
            continue

        for index, line in enumerate(lines):
            match = re.match(r'^\s*names\s*:\s*(.*)$', line)
            if not match:
                continue
            value = match.group(1).strip()
            if value:
                try:
                    parsed_names = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(parsed_names, dict):
                    return [
                        str(parsed_names[key])
                        for key in sorted(parsed_names, key=lambda key: int(key))
                    ]
                if isinstance(parsed_names, (list, tuple)):
                    return [str(name) for name in parsed_names]

            block_names = []
            for item in lines[index + 1:]:
                if item.strip() and not item.lstrip().startswith('-') and not re.match(r'^\s+\d+\s*:', item):
                    break
                list_item = re.match(r'^\s*-\s*(.*?)\s*$', item)
                map_item = re.match(r'^\s*\d+\s*:\s*(.*?)\s*$', item)
                if list_item or map_item:
                    raw_name = (list_item or map_item).group(1).strip()
                    block_names.append(raw_name.strip('"\''))
            if block_names:
                return block_names
    return []


from app.database.db_manager import DatabaseManager
from app.hardware.illumination import IlluminationUnavailable
from app.health.marker_engine import MarkerEngine
from app.health.condition_engine import ConditionEngine
from app.image_processing import (
    StripDetector, StripNotFound, PadDetector, ColorAnalyzer,
    WhiteReferenceNormalizer, QualityGateChecker
)
from app.ml.saliva_ph import predict_saliva_ph

# Phase 1: Lab baseline predictor (replaces RGB Euclidean + grayscale fallback)
try:
    from app.ml.baseline import predict_level as _baseline_predict
    _BASELINE_AVAILABLE = True
except ImportError:
    _BASELINE_AVAILABLE = False
    _baseline_predict = None

# Phase 5/6: ML model inference (active model per strip/marker)
try:
    from app.ml.features import extract_pad_features, FEATURE_VERSION
    from app.ml.trainer import load_model as _load_ml_model, predict as _ml_predict
    _ML_INFERENCE_AVAILABLE = True
except ImportError:
    _ML_INFERENCE_AVAILABLE = False
    extract_pad_features = None
    _load_ml_model = None
    _ml_predict = None

_ml_model_cache: dict = {}  # {(strip_type, marker, model_id): artifact}

main_bp = Blueprint('main', __name__)

def get_db():
    return current_app.db

def get_camera():
    return current_app.camera

def get_illumination():
    return getattr(current_app, 'illumination', None)

# ========== PAGE ROUTES ==========

@main_bp.route('/')
def dashboard():
    """Main dashboard."""
    db = get_db()
    stats = db.get_stats()
    recent = db.get_recent_assessments(10)
    status_labels = {
        'normal': 'Within range',
        'attention': 'Attention',
        'followup_recommended': 'Follow-up recommended',
        'invalid': 'Invalid result',
        'pending': 'Pending'
    }
    status_order = tuple(status_labels)
    outcome_summary = [
        {
            'status': status,
            'label': status_labels.get(status, status.replace('_', ' ').title()),
            'count': count
        }
        for status, count in sorted(
            stats['status_counts'].items(),
            key=lambda item: (
                status_order.index(item[0])
                if item[0] in status_labels else len(status_order),
                item[0]
            )
        )
    ]
    return render_template('dashboard.html', 
                          stats=stats, 
                          recent=recent,
                          outcome_summary=outcome_summary,
                          demo_mode=current_app.config.get('DEMO_MODE', True))

@main_bp.route('/new_screening')
def new_screening():
    """Start new screening."""
    db = get_db()
    patient_id = (request.args.get('patient_id') or '').strip()
    selected_patient = db.get_patient(patient_id) if patient_id else None
    if patient_id and not selected_patient:
        return render_template('error.html', error='Patient not found'), 404
    patients = db.get_all_patients() if not selected_patient else []
    screening_profile_order = ('URS-4K', 'SALIVA-PH4')
    profiles_by_id = {
        profile['profile_id']: profile
        for profile in db.get_all_test_profiles()
        if profile.get('profile_id') in screening_profile_order
    }
    test_profiles = [
        profiles_by_id[profile_id]
        for profile_id in screening_profile_order
        if profile_id in profiles_by_id
    ]
    return render_template(
        'new_screening.html',
        patients=patients,
        test_profiles=test_profiles,
        selected_patient=selected_patient
    )

@main_bp.route('/patients')
def patients():
    """Patient management."""
    db = get_db()
    patients = db.get_all_patients()
    return render_template('patients.html', patients=patients)

@main_bp.route('/camera')
def camera_view():
    """Camera view for screening."""
    db = get_db()
    patient_id = request.args.get('patient_id', '')
    test_profile = (request.args.get('test_profile') or '').strip()
    if not test_profile:
        return render_template('error.html', error='Select a test profile before capturing a photo'), 400
    profile = db.get_test_profile(test_profile)
    if not profile:
        return render_template('error.html', error=f'Test profile {test_profile} not found'), 404
    camera = get_camera()
    if camera and not camera.is_initialized:
        camera.initialize()
    sample_type = profile.get('sample_type', 'saliva')
    patient = db.get_patient(patient_id) if patient_id else None
    return render_template('camera.html', 
                          patient=patient, 
                          sample_type=sample_type, 
                          test_profile=test_profile,
                          test_profile_name=profile.get('name', test_profile),
                          test_markers=profile.get('configuration', {}).get('markers', []))

@main_bp.route('/results/<assessment_id>')
def results(assessment_id):
    """View assessment results."""
    db = get_db()
    assessment = db.get_assessment(assessment_id)
    if not assessment:
        return render_template('error.html', error='Assessment not found'), 404
    patient = db.get_patient(assessment.get('patient_id'))
    return render_template(
        'results.html',
        assessment=assessment,
        patient=patient,
        has_source_image=bool(_assessment_image_path(assessment, 'source')),
        has_analysis_overlay=bool(_assessment_image_path(assessment, 'overlay'))
    )


def _assessment_image_path(assessment, kind):
    """Return an assessment image only when it resolves inside the upload directory."""
    if kind not in {'source', 'overlay'}:
        return None
    source_path = assessment.get('image_path')
    if not source_path:
        return None

    upload_root = os.path.realpath(current_app.config.get('UPLOAD_DIR', 'uploads'))
    source_path = os.path.realpath(source_path)
    try:
        if os.path.commonpath([upload_root, source_path]) != upload_root:
            return None
    except ValueError:
        return None

    target_path = source_path
    if kind == 'overlay':
        target_path = os.path.join(
            os.path.dirname(source_path),
            f"overlay_{os.path.basename(source_path)}"
        )
        target_path = os.path.realpath(target_path)
        try:
            if os.path.commonpath([upload_root, target_path]) != upload_root:
                return None
        except ValueError:
            return None

    return target_path if os.path.isfile(target_path) else None


@main_bp.route('/assessments/<assessment_id>/image/<kind>')
def assessment_image(assessment_id, kind):
    """Serve only source/overlay images belonging to an assessment in UPLOAD_DIR."""
    assessment = get_db().get_assessment(assessment_id)
    if not assessment:
        return jsonify({'error': 'Assessment not found'}), 404
    image_path = _assessment_image_path(assessment, kind)
    if not image_path:
        return jsonify({'error': 'Assessment image not found'}), 404
    return send_file(image_path, conditional=True)

@main_bp.route('/history/<patient_id>')
def history(patient_id):
    """View patient history."""
    db = get_db()
    patient = db.get_patient(patient_id)
    if not patient:
        return render_template('error.html', error='Patient not found'), 404
    assessments = db.get_patient_assessments(patient_id)
    assessment_images = {}
    for assessment in assessments:
        assessment_id = assessment.get('assessment_id')
        assessment_images[assessment_id] = {
            'source': bool(_assessment_image_path(assessment, 'source')),
            'overlay': bool(_assessment_image_path(assessment, 'overlay')),
        }
        for marker_result in assessment.get('marker_results', []):
            confidence = marker_result.get('confidence')
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
            marker_result['confidence_percent'] = (
                round(confidence * 100)
                if confidence is not None and math.isfinite(confidence) and 0 <= confidence <= 1
                else None
            )
            similarity = marker_result.get('color_similarity_percent')
            try:
                similarity = int(similarity)
            except (TypeError, ValueError):
                similarity = None
            marker_result['color_similarity_percent'] = (
                similarity if similarity is not None and 0 <= similarity <= 100 else None
            )

    history_markers = list(dict.fromkeys(
        marker_result.get('marker')
        for assessment in assessments
        for marker_result in assessment.get('marker_results', [])
        if marker_result.get('marker')
    ))
    return render_template(
        'history.html',
        patient=patient,
        assessments=assessments,
        history_markers=history_markers,
        assessment_images=assessment_images
    )

@main_bp.route('/health_insights')
def health_insights():
    """Health insights dashboard."""
    db = get_db()
    stats = db.get_stats()
    recent = db.get_recent_assessments(20)
    return render_template('health_insights.html', stats=stats, recent=recent)

@main_bp.route('/system')
def system():
    """System settings and status."""
    db = get_db()
    camera = get_camera()
    camera_info = camera.get_camera_info() if camera else None
    audit_log = db.get_audit_log(20)
    return render_template('system.html', 
                          camera_info=camera_info, 
                          audit_log=audit_log, 
                          config=current_app.config)

@main_bp.route('/ml_studio')
def ml_studio():
    """ML Studio — dataset management, training, model activation."""
    db = get_db()
    datasets = db.list_datasets()
    yolo_root = os.path.join(
        current_app.config.get('UPLOAD_DIR', 'uploads'), 'yolo_datasets'
    )
    for dataset in datasets:
        dataset['yolo_ready'] = os.path.isfile(
            os.path.join(yolo_root, dataset['dataset_id'], 'data.yaml')
        )
    models   = db.list_models()
    jobs     = db.list_training_jobs(limit=20)
    return render_template(
        'ml_studio.html',
        datasets=datasets,
        models=models,
        jobs=jobs,
        config=current_app.config
    )

# ========== API ROUTES ==========

@main_bp.route('/api/patients/search')
def search_patients():
    """Search patients API."""
    query = request.args.get('q', '')
    db = get_db()
    patients = db.search_patients(query)
    return jsonify(patients)

@main_bp.route('/api/patients/create', methods=['POST'])
def create_patient():
    """Create patient API - FULL DATA."""
    data = request.json
    db = get_db()
    
    # Validate required fields
    if not data.get('first_name') or not data.get('last_name'):
        return jsonify({'error': 'First name and last name are required'}), 400
    
    patient_id = db.create_patient(data)
    return jsonify({'patient_id': patient_id, 'success': True})

@main_bp.route('/api/patients/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update patient API - FULL DATA."""
    data = request.json
    db = get_db()
    
    # Validate required fields
    if not data.get('first_name') or not data.get('last_name'):
        return jsonify({'error': 'First name and last name are required'}), 400
    
    success = db.update_patient(patient_id, data)
    return jsonify({'success': success})

@main_bp.route('/api/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient, their assessments, and safely-contained uploaded images."""
    db = get_db()
    try:
        assessments = db.get_patient_assessments(patient_id)
        if not db.get_patient(patient_id):
            return jsonify({'success': False, 'error': 'Patient not found'}), 404

        upload_root = os.path.realpath(current_app.config.get('UPLOAD_DIR', 'uploads'))
        files_to_remove = []
        for assessment in assessments:
            source = _assessment_image_path(assessment, 'source')
            overlay = _assessment_image_path(assessment, 'overlay')
            files_to_remove.extend(path for path in (source, overlay) if path)

        if not db.delete_patient(patient_id):
            return jsonify({'success': False, 'error': 'Patient could not be deleted'}), 409
    except sqlite3.DatabaseError:
        logger.exception("Patient and assessment deletion failed for %s", patient_id)
        return jsonify({
            'success': False,
            'error': 'Patient and screening history could not be deleted; records were retained'
        }), 500

    cleanup_warnings = []
    for path in files_to_remove:
        try:
            if os.path.commonpath([upload_root, os.path.realpath(path)]) == upload_root:
                os.remove(path)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            logger.warning("Could not remove assessment image %s: %s", path, exc)
            cleanup_warnings.append(os.path.basename(path))

    return jsonify({
        'success': True,
        'deleted_assessments': len(assessments),
        'cleanup_warnings': cleanup_warnings
    })


@main_bp.route(
    '/api/patients/<patient_id>/assessments/<assessment_id>',
    methods=['DELETE']
)
def delete_patient_assessment(patient_id, assessment_id):
    """Delete one screening result and its safely-contained image files."""
    db = get_db()
    if not db.get_patient(patient_id):
        return jsonify({'success': False, 'error': 'Patient not found'}), 404

    assessment = db.get_assessment(assessment_id)
    if not assessment or assessment.get('patient_id') != patient_id:
        return jsonify({'success': False, 'error': 'Screening result not found for this patient'}), 404

    image_paths = [
        path for path in (
            _assessment_image_path(assessment, 'source'),
            _assessment_image_path(assessment, 'overlay')
        )
        if path
    ]

    try:
        deleted = db.delete_assessment(patient_id, assessment_id)
    except sqlite3.DatabaseError:
        logger.exception(
            "Could not delete assessment %s for patient %s",
            assessment_id,
            patient_id
        )
        return jsonify({
            'success': False,
            'error': 'Screening result could not be deleted; it was retained'
        }), 500

    if not deleted:
        return jsonify({'success': False, 'error': 'Screening result could not be deleted'}), 409

    upload_root = os.path.realpath(current_app.config.get('UPLOAD_DIR', 'uploads'))
    cleanup_warnings = []
    for path in image_paths:
        try:
            if os.path.commonpath([upload_root, os.path.realpath(path)]) == upload_root:
                os.remove(path)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            logger.warning("Could not remove assessment image %s: %s", path, exc)
            cleanup_warnings.append(os.path.basename(path))

    return jsonify({'success': True, 'cleanup_warnings': cleanup_warnings})

@main_bp.route('/api/camera/preview')
def camera_preview():
    """Get camera preview frame."""
    camera = get_camera()
    if not camera:
        return jsonify({'error': 'Camera not available'}), 503

    if not camera.is_initialized:
        return jsonify({'error': 'Camera not available'}), 503
    
    try:
        frame = camera.get_live_preview()
        if frame is None:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({'error': 'Failed to encode image'}), 500
        
        img_io = BytesIO(buffer.tobytes())
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/camera/stream')
def camera_stream():
    """Stream camera preview frames as MJPEG instead of polling per frame."""
    camera = get_camera()
    if not camera or not camera.is_initialized:
        return jsonify({'error': 'Camera not available'}), 503

    @stream_with_context
    def generate():
        while camera.is_initialized:
            frame = camera.get_live_preview()
            if frame is None:
                logger.warning("Camera preview stream ended because no frame was captured")
                break

            encoded, buffer = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not encoded:
                logger.warning("Camera preview stream ended because frame encoding failed")
                break

            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                + f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('ascii')
                + frame_bytes
                + b'\r\n'
            )

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'X-Accel-Buffering': 'no',
        },
    )

@main_bp.route('/api/camera/focus', methods=['POST'])
def set_camera_focus():
    """Set manual focus for cameras that expose focus controls."""
    camera = get_camera()
    if not camera or not camera.is_initialized:
        return jsonify({'success': False, 'error': 'Camera not available'}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'A JSON focus value is required'}), 400

    try:
        focus_value = float(data.get('value'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Focus must be a number from 0 to 255'}), 400
    if not math.isfinite(focus_value) or not 0 <= focus_value <= 255:
        return jsonify({'success': False, 'error': 'Focus must be a number from 0 to 255'}), 400

    try:
        result = camera.set_manual_focus(focus_value)
    except RuntimeError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    if not result.get('supported'):
        return jsonify({
            'success': False,
            'error': 'This camera or driver does not support manual focus.',
            **result,
        }), 422
    return jsonify({'success': True, **result})

@main_bp.route('/api/illumination/brightness', methods=['GET', 'POST'])
def control_illumination():
    """Read or set strip-light brightness through Raspberry Pi GPIO."""
    controller = get_illumination()
    if controller is None:
        return jsonify({
            'success': False,
            'available': False,
            'error': 'GPIO illumination is not configured for this server.',
        }), 503

    if request.method == 'GET':
        status = controller.status()
        return jsonify({'success': status['available'], **status}), (
            200 if status['available'] else 503
        )

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'A JSON brightness value is required'}), 400

    value = data.get('value')
    if isinstance(value, bool):
        return jsonify({'success': False, 'error': 'Brightness must be a number from 0 to 100'}), 400
    try:
        percentage = float(value)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Brightness must be a number from 0 to 100'}), 400
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        return jsonify({'success': False, 'error': 'Brightness must be a number from 0 to 100'}), 400

    try:
        result = controller.set_brightness(percentage)
    except IlluminationUnavailable as exc:
        return jsonify({'success': False, 'available': False, 'error': str(exc)}), 503
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    return jsonify({'success': True, **result})

@main_bp.route('/api/camera/capture', methods=['POST'])
def capture_image():
    """Capture image for analysis."""
    camera = get_camera()
    if not camera:
        return jsonify({'error': 'Camera not available'}), 503

    if not camera.is_initialized:
        return jsonify({'error': 'Camera not available'}), 503
    
    frame = camera.capture_frame(timeout=5.0)
    if frame is None:
        return jsonify({'error': 'No frame captured'}), 500

    # Match the live preview's horizontal and vertical mirror for captured photos.
    frame = cv2.flip(frame, -1)
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(current_app.config['UPLOAD_DIR'], filename)
    cv2.imwrite(filepath, frame)
    
    return jsonify({
        'success': True,
        'image_path': filepath,
        'quality': {'pass': True, 'score': 85, 'issues': []}
    })

@main_bp.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Release the camera when the capture page is no longer in use."""
    camera = get_camera()
    if camera and camera.is_initialized:
        camera.release()
    return jsonify({'success': True})

@main_bp.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze captured image using CIE Lab baseline predictor."""
    data = request.get_json(silent=True) or request.form
    patient_id = (data.get('patient_id') or '').strip()
    test_profile = (data.get('test_profile') or '').strip()
    if not test_profile:
        return jsonify({
            'status': 'error',
            'message': 'test_profile is required; select a strip before analysis'
        }), 400
    db = get_db()

    if not patient_id:
        return jsonify({'status': 'error', 'message': 'patient_id is required'}), 400

    profile = db.get_test_profile(test_profile)
    if not profile:
        return jsonify({'status': 'error', 'message': f'Test profile {test_profile} not found'}), 404

    markers = profile.get('configuration', {}).get('markers', [])
    if (not markers or any(not isinstance(marker, str) or not marker.strip() for marker in markers)
            or len(set(markers)) != len(markers)):
        return jsonify({'status': 'error', 'message': f'Test profile {test_profile} has an invalid marker configuration'}), 400

    sample_type = profile.get('sample_type', 'saliva')
    image_upload = request.files.get('image')
    image_path = data.get('image_path')
    if image_upload:
        from werkzeug.utils import secure_filename
        original_name = secure_filename(image_upload.filename or '')
        extension = os.path.splitext(original_name)[1].lower()
        if extension not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            return jsonify({'status': 'error', 'message': 'Upload a JPG, PNG, WEBP, or BMP image'}), 400
        filename = f"upload_{uuid.uuid4().hex}{extension}"
        image_path = os.path.join(current_app.config['UPLOAD_DIR'], filename)
        image_upload.save(image_path)

    if not image_path or not os.path.exists(image_path):
        return jsonify({'status': 'error', 'message': 'Image not found'}), 404

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'status': 'error', 'message': 'Failed to load image'}), 500

    # Initialize processors (check for active YOLO strip detector)
    yolo_detector = None
    yolo_class_names = []
    try:
        yolo_row = db.get_active_model(test_profile, 'strip_detector')
        if not yolo_row:
            yolo_row = db.get_active_model('URS-4K', 'strip_detector')
        if yolo_row and os.path.exists(yolo_row.get('model_path', '')):
            yolo_class_names = _read_yolo_class_names(
                yolo_row, current_app.config.get('UPLOAD_DIR', 'uploads')
            )
            if not yolo_class_names:
                logger.warning(
                    "No YOLO class names found for active model %s; pad class mapping is unavailable",
                    yolo_row.get('model_id'),
                )
            from app.image_processing.yolo_detector import ONNXYOLODetector
            yolo_detector = ONNXYOLODetector(
                yolo_row['model_path'],
                conf_threshold=0.20,
                class_names=yolo_class_names,
            )
    except Exception as exc:
        logger.warning(f"Could not load active YOLO strip detector: {exc}")

    strip_detector = StripDetector(yolo_detector=yolo_detector)
    pad_detector = PadDetector()
    quality_checker = QualityGateChecker()
    white_normalizer = WhiteReferenceNormalizer()

    # 1. Global image quality gate (blur, exposure, glare)
    quality_report = quality_checker.evaluate_image(image)
    if quality_report['retake_required']:
        logger.warning(f"Image quality gate failed: {quality_report['reason']}")
        return jsonify({
            'status': 'retake',
            'method': 'none',
            'reason': f"Image quality check failed: {quality_report['reason']} Please retake photo with better focus/lighting.",
            'quality': {
                'blur': float(quality_report['blur']),
                'exposure': quality_report['exposure'],
                'glare': bool(quality_report['glare']),
                'warnings': quality_report['warnings']
            },
            'detection': {
                'confidence': 0.0,
                'overlay_image': None
            },
            'results': [],
            'disclaimer': 'Screening aid only. Not a diagnosis.',
            'success': False,
            'message': quality_report['reason']
        }), 200

    manual_corners = data.get('manual_corners')

    # 2. Detect strip
    try:
        strip_result = strip_detector.detect(image, manual_corners=manual_corners)
    except StripNotFound as exc:
        logger.warning(f"StripNotFound in analyze: {exc}")
        return jsonify({
            'status': 'retake',
            'method': 'none',
            'reason': f"Test strip could not be detected: {exc}. Please adjust lighting or manually indicate corners.",
            'quality': {
                'blur': float(quality_report['blur']),
                'exposure': quality_report['exposure'],
                'glare': bool(quality_report['glare']),
                'warnings': quality_report['warnings'] + ['Strip not detected']
            },
            'detection': {
                'confidence': 0.0,
                'overlay_image': None
            },
            'results': [],
            'disclaimer': 'Screening aid only. Not a diagnosis.',
            'success': False,
            'message': 'Strip not found. Please retake photo.'
        }), 200

    detection_confidence = float(strip_result.get('confidence', 1.0))

    # 3. Strip resolution gate
    res_ok, res_msg = quality_checker.evaluate_strip_resolution(strip_result)
    if not res_ok:
        logger.warning(f"Strip resolution gate failed: {res_msg}")
        return jsonify({
            'status': 'retake',
            'method': 'none',
            'reason': f"Strip resolution check failed: {res_msg}",
            'quality': {
                'blur': float(quality_report['blur']),
                'exposure': quality_report['exposure'],
                'glare': bool(quality_report['glare']),
                'warnings': quality_report['warnings'] + [res_msg]
            },
            'detection': {
                'confidence': detection_confidence,
                'overlay_image': None
            },
            'results': [],
            'disclaimer': 'Screening aid only. Not a diagnosis.',
            'success': False,
            'message': res_msg
        }), 200

    # 4. Detect pads
    is_saliva_ph4 = test_profile == 'SALIVA-PH4'
    configured_pad_count = profile.get('pad_count') or len(markers)
    pad_count = 4 if is_saliva_ph4 else configured_pad_count
    # Saliva pad spacing varies by strip, so locate its colored regions directly.
    pads = pad_detector.detect_pads(
        image,
        strip_result,
        num_pads=pad_count,
        class_names=yolo_class_names,
        marker_names=None if is_saliva_ph4 else markers,
        detect_color_regions=is_saliva_ph4,
    )
    if not pads or (is_saliva_ph4 and len(pads) != 4):
        reason = (
            'All four pads must be visible in the photo. Please retake it with the complete strip in frame.'
            if is_saliva_ph4 else
            'Pads could not be localized on the strip. Please retake photo.'
        )
        return jsonify({
            'status': 'retake',
            'method': 'none',
            'reason': reason,
            'quality': {
                'blur': float(quality_report['blur']),
                'exposure': quality_report['exposure'],
                'glare': bool(quality_report['glare']),
                'warnings': quality_report['warnings'] + ['Pads not localized']
            },
            'detection': {
                'confidence': detection_confidence,
                'overlay_image': None
            },
            'results': [],
            'disclaimer': 'Screening aid only. Not a diagnosis.',
            'success': False,
            'message': reason
        }), 200

    # 5. White Reference Normalization
    rectified_strip = strip_result.get('rectified_strip')
    white_ref_meta = {}
    if rectified_strip is not None:
        pads, white_ref_meta = white_normalizer.normalize_strip_pads(rectified_strip, pads)

    # Generate and save debug overlay image
    overlay_rel_path = None
    overlay_img = None
    try:
        overlay_img = pad_detector.generate_debug_overlay(image, strip_result, pads)
        overlay_filename = f"overlay_{os.path.basename(image_path)}"
        overlay_full_path = os.path.join(os.path.dirname(image_path), overlay_filename)
        if cv2.imwrite(overlay_full_path, overlay_img):
            overlay_rel_path = f"/uploads/{overlay_filename}"
        else:
            logger.warning("Failed to save debug overlay image: %s", overlay_full_path)
    except Exception as overlay_err:
        logger.warning(f"Failed to generate debug overlay: {overlay_err}")

    # Analyse each marker — use active ML model if available, else baseline
    marker_results = []
    active_model_id   = None
    active_model_ver  = None

    if is_saliva_ph4:
        measured_pad_colors = [
            pad.get('color_raw', pad.get('color', [128, 128, 128]))
            for pad in pads
        ]
        pH_match = predict_saliva_ph(measured_pad_colors)
        measured_average = np.mean(measured_pad_colors, axis=0).round().astype(int).tolist()
        chart_average = np.mean(
            [comparison['chart_rgb'] for comparison in pH_match['pad_comparisons']],
            axis=0
        ).round().astype(int).tolist()
        marker_results.append({
            'marker': 'pH',
            'pad_index': None,
            'result': pH_match['level'],
            'level': pH_match['level'],
            'match_quality': 'low' if pH_match['low_confidence'] else 'good',
            'confidence': pH_match['confidence'],
            'delta_e_best': pH_match['delta_e_best'],
            'delta_e_second': pH_match['delta_e_second'],
            'chart_match_level': pH_match['level'],
            'chart_match_delta_e': pH_match['delta_e_best'],
            'chart_match_rgb': chart_average,
            'color_similarity_percent': pH_match['color_similarity_percent'],
            'low_confidence': pH_match['low_confidence'],
            'color': measured_average,
            'color_raw': measured_average,
            'status': 'research',
            'method': 'four_pad_chart',
            'pad_comparisons': pH_match['pad_comparisons'],
            'reversed_chart_order': pH_match['reversed_chart_order'],
        })

    markers_to_process = [] if is_saliva_ph4 else markers
    for i, marker in enumerate(markers_to_process):
        pad = next(
            (candidate for candidate in pads if candidate.get('marker') == marker),
            pads[i] if i < len(pads) and not any('marker' in candidate for candidate in pads) else None,
        )
        color = np.array(pad.get('color', [128, 128, 128])) if pad else np.array([128, 128, 128])
        white_ref_rgb = white_ref_meta.get('white_ref_rgb') if white_ref_meta else None

        ref_colors = db.get_reference_colors(test_profile, marker)
        if not ref_colors:
            marker_results.append({
                'marker': marker, 'pad_index': i, 'result': 'no_reference',
                'level': 'no_reference', 'confidence': 0.0,
                'delta_e_best': None, 'delta_e_second': None, 'low_confidence': True,
                'color': color.tolist(), 'status': 'error',
                'error': f'No reference colors found for {test_profile}/{marker}',
                'method': 'none',
            })
            continue

        level = 'unknown'; confidence = 0.0; de_best = None; de_second = None
        low_conf = True; method = 'baseline'
        chart_match = None
        chart_match_rgb = None
        if _BASELINE_AVAILABLE:
            try:
                chart_match = _baseline_predict(color.tolist(), ref_colors)
                matched_reference = next(
                    (
                        reference for reference in ref_colors
                        if reference.get('level') == chart_match['level']
                    ),
                    None,
                )
                if matched_reference:
                    reference_rgb = matched_reference.get('rgb')
                    if (
                        isinstance(reference_rgb, (list, tuple))
                        and len(reference_rgb) == 3
                    ):
                        chart_match_rgb = [
                            int(max(0, min(255, channel)))
                            for channel in reference_rgb
                        ]
            except Exception as exc:
                logger.warning("Chart color comparison failed for %s: %s", marker, exc)

        # -- Try active ML model first --
        ml_used = False
        if _ML_INFERENCE_AVAILABLE:
            try:
                ml_model_row = db.get_active_model(test_profile, marker)
                if ml_model_row:
                    cache_key = (test_profile, marker, ml_model_row['model_id'])
                    if cache_key not in _ml_model_cache:
                        artifact = _load_ml_model(
                            ml_model_row['model_path'], ml_model_row['sha256_hash']
                        )
                        # Check feature version compatibility
                        if artifact.get('feature_version', -1) != FEATURE_VERSION:
                            raise ValueError(
                                f"Feature version mismatch: model={artifact.get('feature_version')}, "
                                f"code={FEATURE_VERSION}"
                            )
                        _ml_model_cache[cache_key] = artifact
                    artifact = _ml_model_cache[cache_key]
                    feat = extract_pad_features(
                        color.tolist(),
                        white_ref_rgb=white_ref_rgb
                    )
                    level, confidence = _ml_predict(artifact, feat)
                    low_conf = confidence < 0.60
                    de_best = None; de_second = None
                    method = 'ml'
                    active_model_id  = ml_model_row['model_id']
                    active_model_ver = ml_model_row['version']
                    ml_used = True
            except Exception as ml_exc:
                logger.warning(f"ML inference failed for {marker}, falling back to baseline: {ml_exc}")

        # -- Baseline fallback --
        if not ml_used:
            if chart_match is not None:
                level = chart_match['level']; confidence = chart_match['confidence']
                de_best = chart_match['delta_e_best']; de_second = chart_match['delta_e_second']
                low_conf = chart_match['low_confidence']; method = 'baseline'
            elif _BASELINE_AVAILABLE:
                level = 'unknown'; confidence = 0.0; de_best = None
                de_second = None; low_conf = True; method = 'error'
            else:
                color_analyzer = ColorAnalyzer()
                old_result = color_analyzer.analyze_marker_color(marker, color, {marker: ref_colors})
                level = old_result.get('result', 'unknown'); confidence = old_result.get('confidence', 0.5)
                de_best = None; de_second = None; low_conf = confidence < 0.5; method = 'rgb_legacy'

        if level in ['+1', '+2', '+3', '+4', 'positive', 'abnormal', 'trace']:
            status = 'attention'
        else:
            status = 'normal'

        marker_results.append({
            'marker': marker,
            'pad_index': i,
            'result': level,
            'level': level,
            'match_quality': 'low' if low_conf else 'good',
            'confidence': confidence,
            'delta_e_best': de_best,
            'delta_e_second': de_second,
            'chart_match_level': chart_match['level'] if chart_match else None,
            'chart_match_delta_e': chart_match['delta_e_best'] if chart_match else None,
            'chart_match_rgb': chart_match_rgb,
            'color_similarity_percent': (
                chart_match['color_similarity_percent'] if chart_match else None
            ),
            'low_confidence': low_conf,
            'color': color.tolist(),
            'color_raw': (
                pad.get('color_raw', pad.get('color', [128, 128, 128]))
                if pad else [128, 128, 128]
            ),
            'status': status,
            'method': method,
        })

    if overlay_img is not None and overlay_rel_path:
        try:
            line_height = 24
            footer_height = line_height * (len(marker_results) + 1)
            footer = np.full((footer_height, overlay_img.shape[1], 3), 32, dtype=np.uint8)
            footer_title = (
                'Combined four-pad saliva pH estimate:'
                if is_saliva_ph4 else
                'Predicted readings by detected pad index:'
            )
            cv2.putText(
                footer, footer_title,
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            for index, result in enumerate(marker_results):
                marker_name = result.get('marker', 'Unknown').replace('_', ' ').title()
                if marker_name == 'Ph':
                    marker_name = 'pH'
                label = (
                    f"{marker_name}: {result.get('result', 'unknown')}"
                    if is_saliva_ph4 else
                    f"#{index} {marker_name}: {result.get('result', 'unknown')}"
                )
                cv2.putText(
                    footer, label,
                    (10, line_height * (index + 1) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
            annotated_overlay = cv2.vconcat([overlay_img, footer])
            overlay_filename = f"overlay_{os.path.basename(image_path)}"
            overlay_full_path = os.path.join(os.path.dirname(image_path), overlay_filename)
            if not cv2.imwrite(overlay_full_path, annotated_overlay):
                logger.warning("Failed to save predicted-reading overlay: %s", overlay_full_path)
                overlay_rel_path = None
        except Exception as overlay_err:
            logger.warning("Failed to add predicted readings to debug overlay: %s", overlay_err)
            overlay_rel_path = None

    # Interpret markers and generate clinical associations
    marker_engine = MarkerEngine(db, sample_type=sample_type)
    condition_engine = ConditionEngine(db)

    if is_saliva_ph4:
        interpreted = [{
            **marker_results[0],
            'interpretation': 'Approximate color-chart match for research use only.',
        }]
        associations = []
        summary = {
            'overall_status': 'research',
            'flagged_markers': [],
            'health_domains': {},
        }
        recommendations = [
            'This color-chart estimate is not a diagnosis or clinical test.'
        ]
        limitations = [
            'Research use only; the four-pad reference colors were sampled from supplied product images and are not calibrated or clinically validated.',
            'Lighting, camera settings, strip orientation, and image processing can affect color matching.',
        ]
    else:
        interpreted = marker_engine.interpret_all_markers(marker_results)
        results_by_marker = {r['marker']: r for r in marker_results}
        for item in interpreted:
            m = item.get('marker')
            if m in results_by_marker:
                orig = results_by_marker[m]
                item['level'] = orig.get('level', item.get('result'))
                item['pad_index'] = orig.get('pad_index')
                item['delta_e_best'] = orig.get('delta_e_best')
                item['delta_e_second'] = orig.get('delta_e_second')
                item['chart_match_level'] = orig.get('chart_match_level')
                item['chart_match_delta_e'] = orig.get('chart_match_delta_e')
                item['chart_match_rgb'] = orig.get('chart_match_rgb')
                item['color_similarity_percent'] = orig.get('color_similarity_percent')
                item['low_confidence'] = orig.get('low_confidence', False)
                item['color'] = orig.get('color')
                item['color_raw'] = orig.get('color_raw', orig.get('color'))
                item['method'] = orig.get('method', 'baseline')
                item['pad_comparisons'] = orig.get('pad_comparisons')

        associations = condition_engine.find_associations(interpreted)
        summary = marker_engine.generate_summary(marker_results)
        recommendations = condition_engine.get_recommendations(associations)
        limitations = condition_engine.get_limitations(associations)

    if sample_type == 'saliva' and not is_saliva_ph4:
        limitations.append("RESEARCH USE ONLY — Saliva results require clinical validation")
        recommendations.append("Consult with a healthcare professional for clinical interpretation")

    assessment_data = {
        'patient_id': patient_id,
        'sample_type': sample_type,
        'test_profile': test_profile,
        'overall_status': summary.get('overall_status', 'normal'),
        'image_quality': {'overall': 'good'},
        'analysis_quality': {'overall': 'good', 'method': strip_result.get('method', 'baseline')},
        'marker_results': interpreted,
        'flagged_markers': summary.get('flagged_markers', []),
        'health_domains': summary.get('health_domains', {}),
        'possible_associations': associations,
        'recommendations': recommendations,
        'limitations': limitations,
        'evidence_levels': condition_engine.get_evidence_summary(associations),
        'image_path': image_path
    }

    assessment_id = db.create_assessment(assessment_data)
    db.log_action('system', 'analysis_completed', f'Assessment {assessment_id} created ({strip_result.get("method", "baseline")})', assessment_id)

    return jsonify({
        'status': 'ok',
        'method': strip_result.get('method', 'ml' if active_model_id else 'baseline'),
        'assessment_id': assessment_id,
        'strip_type': test_profile,
        'quality': {
            'blur': float(quality_report['blur']),
            'exposure': quality_report['exposure'],
            'glare': bool(quality_report['glare']),
            'warnings': quality_report['warnings'],
            'white_reference': white_ref_meta
        },
        'detection': {
            'confidence': detection_confidence,
            'method': strip_result.get('method', 'cv_contour'),
            'overlay_image': overlay_rel_path
        },
        'results': interpreted,
        'model': {'id': active_model_id, 'version': active_model_ver},
        'disclaimer': 'Screening aid only. Not a diagnosis.',
        # Legacy fields for backward-compat with existing frontend:
        'success': True,
        'markers': interpreted,
        'summary': summary,
        'associations': associations,
        'recommendations': recommendations,
        'limitations': limitations
    })

@main_bp.route('/api/system/status')
def system_status():
    """System status API."""
    camera = get_camera()
    db = get_db()
    status = {
        'camera': {
            'status': 'online' if camera and camera.is_initialized else 'offline',
            'info': camera.get_camera_info() if camera else None
        },
        'database': {'status': 'ready', 'stats': db.get_stats()},
        'system': {
            'time': datetime.now().isoformat(), 
            'demo_mode': current_app.config.get('DEMO_MODE', True),
            'sample_type': 'saliva'
        }
    }
    return jsonify(status)

@main_bp.route('/api/system/demo_data', methods=['POST'])
def generate_demo_data():
    """Generate demo data for presentation."""
    db = get_db()
    
    demo_patients = [
        {'first_name': 'John', 'last_name': 'Smith', 'date_of_birth': '1985-06-15', 'gender': 'Male', 'email': 'john@example.com', 'phone': '+1 555-0101', 'address': '123 Main St, NY'},
        {'first_name': 'Jane', 'last_name': 'Doe', 'date_of_birth': '1990-03-22', 'gender': 'Female', 'email': 'jane@example.com', 'phone': '+1 555-0102', 'address': '456 Oak Ave, CA'},
        {'first_name': 'Robert', 'last_name': 'Johnson', 'date_of_birth': '1978-11-08', 'gender': 'Male', 'email': 'robert@example.com', 'phone': '+1 555-0103', 'address': '789 Pine St, TX'},
        {'first_name': 'Maria', 'last_name': 'Garcia', 'date_of_birth': '1995-08-30', 'gender': 'Female', 'email': 'maria@example.com', 'phone': '+1 555-0104', 'address': '321 Elm St, FL'},
        {'first_name': 'David', 'last_name': 'Brown', 'date_of_birth': '1982-04-12', 'gender': 'Male', 'email': 'david@example.com', 'phone': '+1 555-0105', 'address': '654 Cedar St, WA'},
    ]
    
    patient_ids = []
    for patient_data in demo_patients:
        pid = db.create_patient(patient_data)
        patient_ids.append(pid)
    
    statuses = ['normal', 'normal', 'attention', 'normal', 'followup_recommended', 'normal']
    for i, pid in enumerate(patient_ids):
        for j in range(3):
            status = statuses[(i + j) % len(statuses)]
            assessment_data = {
                'patient_id': pid,
                'sample_type': 'saliva',
                'test_profile': 'SALIVA-4K',
                'overall_status': status,
                'marker_results': [
                    {'marker': 'glucose', 'result': 'negative', 'status': 'normal'},
                    {'marker': 'protein', 'result': 'negative' if status == 'normal' else 'trace', 'status': status},
                    {'marker': 'ketones', 'result': 'negative', 'status': 'normal'},
                    {'marker': 'blood', 'result': 'negative', 'status': 'normal'},
                    {'marker': 'nitrite', 'result': 'negative', 'status': 'normal'},
                    {'marker': 'leukocytes', 'result': 'negative' if status == 'normal' else '+1', 'status': status},
                    {'marker': 'pH', 'result': '7.0', 'status': 'normal'},
                    {'marker': 'specific_gravity', 'result': '1.010', 'status': 'normal'},
                    {'marker': 'bilirubin', 'result': 'negative', 'status': 'normal'},
                    {'marker': 'urobilinogen', 'result': 'normal', 'status': 'normal'},
                ],
                'flagged_markers': [] if status == 'normal' else [{'marker': 'protein', 'result': 'trace'}],
                'recommendations': ['No significant findings'] if status == 'normal' else ['Research findings - consult healthcare provider'],
                'limitations': ['RESEARCH USE ONLY - Saliva results require clinical validation']
            }
            db.create_assessment(assessment_data)
    
    return jsonify({'success': True, 'message': 'Demo data generated with saliva samples'})
