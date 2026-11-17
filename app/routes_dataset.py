"""
LUMINE — Phase 4 Dataset Management Routes
Provides REST API + annotation page for building labeled training datasets.
"""

import csv
import io
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import (
    Blueprint, current_app, jsonify, render_template,
    request, send_file
)

logger = logging.getLogger(__name__)

dataset_bp = Blueprint('dataset', __name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db():
    return current_app.db


def _allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _upload_dir() -> str:
    d = os.path.join(current_app.config.get('UPLOAD_DIR', 'uploads'), 'dataset_samples')
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@dataset_bp.route('/annotation')
def annotation_page():
    """Annotation tool main page."""
    db = _db()
    datasets = db.list_datasets()
    sample_id = request.args.get('sample_id')
    dataset_id = request.args.get('dataset_id')

    sample = None
    labels = []
    profile = None
    markers = []

    if sample_id:
        sample = db.get_sample(sample_id)
        if sample:
            labels = db.get_pad_labels(sample_id)
            profile = db.get_test_profile(sample['strip_type'])
            markers = (profile or {}).get('configuration', {}).get('markers', [])

    return render_template(
        'annotation.html',
        datasets=datasets,
        sample=sample,
        labels=labels,
        profile=profile,
        markers=markers,
        selected_dataset_id=dataset_id,
    )


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all training datasets with sample/label counts."""
    datasets = _db().list_datasets()
    yolo_root = os.path.join(
        current_app.config.get('UPLOAD_DIR', 'uploads'), 'yolo_datasets'
    )
    for dataset in datasets:
        dataset['yolo_ready'] = os.path.isfile(
            os.path.join(yolo_root, dataset['dataset_id'], 'data.yaml')
        )
    return jsonify({"datasets": datasets})


@dataset_bp.route('/api/datasets', methods=['POST'])
def create_dataset():
    """Create a new training dataset."""
    data = request.json or {}
    name = data.get('name', '').strip()
    strip_type = data.get('strip_type', '').strip()
    if not name or not strip_type:
        return jsonify({'error': 'name and strip_type are required'}), 400

    dataset_id = _db().create_dataset(
        name=name,
        strip_type=strip_type,
        description=data.get('description', ''),
        created_by=data.get('created_by', 'user'),
    )
    return jsonify({'dataset_id': dataset_id, 'success': True}), 201


@dataset_bp.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get dataset metadata."""
    ds = _db().get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404
    return jsonify(ds)


@dataset_bp.route('/api/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset and all its samples + labels."""
    ok = _db().delete_dataset(dataset_id)
    return jsonify({'success': ok})


# ---------------------------------------------------------------------------
# Sample operations
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/samples', methods=['GET'])
def list_samples(dataset_id):
    """List samples in a dataset."""
    if not _db().get_dataset(dataset_id):
        return jsonify({'error': 'Dataset not found'}), 404
    return jsonify({"samples": _db().list_samples(dataset_id)})


@dataset_bp.route('/api/datasets/<dataset_id>/samples', methods=['POST'])
def upload_sample(dataset_id):
    """
    Upload a strip photo to a dataset.
    Accepts multipart/form-data with:
      - file: JPEG or PNG image
      - strip_type: (optional override, defaults to dataset strip_type)
      - capture_meta: JSON string with phone/lighting/date info
      - is_synthetic: '1' or 'true'
    """
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'No file selected'}), 400
    if not _allowed_file(f.filename):
        return jsonify({'error': 'Only JPEG and PNG files are accepted'}), 400

    # Read and size-check
    data = f.read()
    if len(data) > MAX_IMAGE_BYTES:
        return jsonify({'error': f'File exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit'}), 413

    # Re-encode through OpenCV to strip EXIF / normalise
    import cv2
    import numpy as np
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    safe_name = f"sample_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(_upload_dir(), safe_name)
    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    strip_type = request.form.get('strip_type') or ds['strip_type']
    is_synthetic = request.form.get('is_synthetic', '').lower() in ('1', 'true', 'yes')
    capture_meta_raw = request.form.get('capture_meta', '{}')
    try:
        capture_meta = json.loads(capture_meta_raw)
    except Exception:
        capture_meta = {}

    sample_id = db.add_sample(
        dataset_id=dataset_id,
        image_path=save_path,
        strip_type=strip_type,
        capture_meta=capture_meta,
        is_synthetic=is_synthetic,
    )
    return jsonify({'sample_id': sample_id, 'success': True, 'image_path': save_path}), 201


@dataset_bp.route('/api/samples/<sample_id>', methods=['GET'])
def get_sample(sample_id):
    """Get sample metadata and its pad labels."""
    db = _db()
    sample = db.get_sample(sample_id)
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    labels = db.get_pad_labels(sample_id)
    return jsonify({'sample': sample, 'labels': labels})


@dataset_bp.route('/api/samples/<sample_id>', methods=['DELETE'])
def delete_sample(sample_id):
    """Delete a sample and all its labels."""
    ok = _db().delete_sample(sample_id)
    return jsonify({'success': ok})


# ---------------------------------------------------------------------------
# Label operations
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/samples/<sample_id>/labels', methods=['GET'])
def get_labels(sample_id):
    """Get all pad labels for a sample."""
    if not _db().get_sample(sample_id):
        return jsonify({'error': 'Sample not found'}), 404
    return jsonify({"labels": _db().get_pad_labels(sample_id)})


@dataset_bp.route('/api/samples/<sample_id>/labels', methods=['POST'])
def save_labels(sample_id):
    """
    Save (upsert) pad labels for a sample.
    Body: {labels: [{marker, pad_index, true_level, measured_value?, label_source?}],
           annotator?: str}
    """
    db = _db()
    if not db.get_sample(sample_id):
        return jsonify({'error': 'Sample not found'}), 404

    body = request.json or {}
    labels = body.get('labels', [])
    annotator = body.get('annotator', 'user')
    if not isinstance(labels, list):
        return jsonify({'error': 'labels must be a list'}), 400

    saved = db.save_pad_labels(sample_id, labels, annotator=annotator)
    return jsonify({'saved': saved, 'success': True})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/export/csv', methods=['GET'])
def export_csv(dataset_id):
    """Export all labelled pads as a CSV file."""
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    rows = db.export_dataset_csv(dataset_id)
    if not rows:
        return jsonify({'error': 'No labelled data in this dataset'}), 404

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    output = io.BytesIO(buf.getvalue().encode('utf-8'))
    output.seek(0)
    safe_ds_name = secure_filename(ds['name']) or dataset_id
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"{safe_ds_name}_labels.csv",
    )


@dataset_bp.route('/api/datasets/<dataset_id>/export/zip', methods=['GET'])
def export_zip(dataset_id):
    """
    Export dataset as ZIP: labels.csv + all sample images.
    Only images that have at least one label are included.
    """
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    rows = db.export_dataset_csv(dataset_id)
    if not rows:
        return jsonify({'error': 'No labelled data in this dataset'}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write CSV
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr('labels.csv', csv_buf.getvalue())

        # Write images (unique paths, if they exist on disk)
        seen_paths = set()
        for row in rows:
            img_path = row.get('image_path', '')
            if img_path and img_path not in seen_paths and os.path.isfile(img_path):
                seen_paths.add(img_path)
                arcname = 'images/' + os.path.basename(img_path)
                zf.write(img_path, arcname)

    buf.seek(0)
    safe_ds_name = secure_filename(ds['name']) or dataset_id
    return send_file(
        buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{safe_ds_name}_dataset.zip",
    )


# ---------------------------------------------------------------------------
# Group split
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/split', methods=['POST'])
def assign_split(dataset_id):
    """
    Assign train/val/test split to dataset samples (group split by photo).
    Body: {train_frac?: float, val_frac?: float}
    """
    db = _db()
    if not db.get_dataset(dataset_id):
        return jsonify({'error': 'Dataset not found'}), 404

    body = request.json or {}
    train_frac = float(body.get('train_frac', 0.7))
    val_frac   = float(body.get('val_frac',   0.15))
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        return jsonify({'error': 'train_frac must be in (0,1); val_frac must be in [0,1)'}), 400
    if train_frac + val_frac >= 1.0:
        return jsonify({'error': 'train_frac + val_frac must be < 1.0'}), 400

    counts = db.assign_group_split(dataset_id, train_frac=train_frac, val_frac=val_frac)
    return jsonify({'split_counts': counts, 'success': True})


# ---------------------------------------------------------------------------
# Bulk upload  (BUG-002 fix)
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/samples/bulk', methods=['POST'])
def bulk_upload_samples(dataset_id):
    """
    Upload multiple strip photos in a single multipart request.
    Fields:
      images      : one or more image files (field name can repeat)
      is_synthetic: '1' or 'true'
      strip_type  : optional override
    Returns: {uploaded: N, failed: [{filename, error}]}
    """
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    files = request.files.getlist('images')
    if not files:
        # Fallback: accept 'file' as well (single-file compat)
        files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No images field in request'}), 400

    strip_type   = request.form.get('strip_type') or ds['strip_type']
    is_synthetic = request.form.get('is_synthetic', '').lower() in ('1', 'true', 'yes')

    import cv2
    import numpy as np

    uploaded = 0
    failed = []
    for f in files:
        if not f or not f.filename:
            continue
        if not _allowed_file(f.filename):
            failed.append({'filename': f.filename, 'error': 'Not a JPEG/PNG'})
            continue
        try:
            data = f.read()
            if len(data) > MAX_IMAGE_BYTES:
                failed.append({'filename': f.filename, 'error': 'File too large (>20 MB)'})
                continue
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                failed.append({'filename': f.filename, 'error': 'Could not decode image'})
                continue
            safe_name = f"sample_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(_upload_dir(), safe_name)
            cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            db.add_sample(
                dataset_id=dataset_id,
                image_path=save_path,
                strip_type=strip_type,
                is_synthetic=is_synthetic,
            )
            uploaded += 1
        except Exception as exc:
            failed.append({'filename': f.filename, 'error': str(exc)})

    return jsonify({'uploaded': uploaded, 'failed': failed, 'success': True}), 201


# ---------------------------------------------------------------------------
# ZIP import  (BUG-004 fix)
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/import/zip', methods=['POST'])
def import_zip(dataset_id):
    """
    Import a ZIP archive containing strip images and an optional labels.csv.

    ZIP layout (flexible):
        images/photo1.jpg
        images/photo2.png
        ...
        labels.csv          (optional)

    labels.csv columns (all optional except filename + true_level):
        filename, marker, pad_index, true_level, label_source, pad_color, white_ref_rgb

    Returns: {uploaded, labelled, failed}
    """
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    if 'zip_file' not in request.files:
        return jsonify({'error': 'No zip_file field in request'}), 400

    zf_upload = request.files['zip_file']
    if not zf_upload.filename.lower().endswith('.zip'):
        return jsonify({'error': 'File must be a .zip archive'}), 400

    strip_type   = request.form.get('strip_type') or ds['strip_type']
    is_synthetic = request.form.get('is_synthetic', '').lower() in ('1', 'true', 'yes')

    import cv2
    import numpy as np

    zip_bytes = zf_upload.read()
    uploaded = 0; labelled = 0; failed = []
    # Map original basename -> sample_id (for label matching)
    basename_to_sid: dict = {}

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()

            # Pass 1: images
            for name in names:
                if name.endswith('/') or '__MACOSX' in name:
                    continue
                base = os.path.basename(name)
                ext = base.rsplit('.', 1)[-1].lower() if '.' in base else ''
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                try:
                    data = zf.read(name)
                    if len(data) > MAX_IMAGE_BYTES:
                        failed.append({'filename': base, 'error': 'File too large'})
                        continue
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        failed.append({'filename': base, 'error': 'Could not decode'})
                        continue
                    safe_name = f"sample_{uuid.uuid4().hex}.jpg"
                    save_path = os.path.join(_upload_dir(), safe_name)
                    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    sid = db.add_sample(
                        dataset_id=dataset_id,
                        image_path=save_path,
                        strip_type=strip_type,
                        is_synthetic=is_synthetic,
                    )
                    basename_to_sid[base] = sid
                    uploaded += 1
                except Exception as exc:
                    failed.append({'filename': base, 'error': str(exc)})

            # Pass 2: labels.csv if present
            csv_names = [n for n in names if os.path.basename(n).lower() == 'labels.csv']
            if csv_names and basename_to_sid:
                import csv as _csv
                raw = zf.read(csv_names[0]).decode('utf-8', errors='replace')
                reader = _csv.DictReader(io.StringIO(raw))
                for row in reader:
                    fname = row.get('filename', '').strip()
                    if not fname:
                        continue
                    sid = basename_to_sid.get(fname)
                    if not sid:
                        # Try matching without extension
                        for b, s in basename_to_sid.items():
                            if b.rsplit('.', 1)[0] == fname.rsplit('.', 1)[0]:
                                sid = s; break
                    if not sid:
                        continue
                    try:
                        db.save_pad_labels(sid, [{
                            'marker':       row.get('marker', ''),
                            'pad_index':    int(row.get('pad_index', 0)),
                            'true_level':   row.get('true_level', ''),
                            'label_source': row.get('label_source', 'csv_import'),
                            'pad_color':    row.get('pad_color'),
                            'white_ref_rgb':row.get('white_ref_rgb'),
                        }])
                        labelled += 1
                    except Exception:
                        pass

    except zipfile.BadZipFile:
        return jsonify({'error': 'Could not open ZIP file — file may be corrupt'}), 400

    return jsonify({'uploaded': uploaded, 'labelled': labelled, 'failed': failed, 'success': True}), 201


# ---------------------------------------------------------------------------
# CSV label import  (BUG-004 fix)
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/import/labels_csv', methods=['POST'])
def import_labels_csv(dataset_id):
    """
    Import a CSV of pad labels for existing samples in this dataset.

    CSV columns (header row required):
        sample_id, marker, pad_index, true_level
        [optional: label_source, pad_color, white_ref_rgb, measured_value]

    Returns: {saved, skipped, errors}
    """
    db = _db()
    if not db.get_dataset(dataset_id):
        return jsonify({'error': 'Dataset not found'}), 404

    if 'csv_file' not in request.files:
        return jsonify({'error': 'No csv_file field in request'}), 400

    import csv as _csv
    f = request.files['csv_file']
    raw = f.read().decode('utf-8', errors='replace')
    reader = _csv.DictReader(io.StringIO(raw))

    saved = 0; skipped = 0; errors = []
    for i, row in enumerate(reader):
        sid  = (row.get('sample_id') or '').strip()
        marker = (row.get('marker') or '').strip()
        level  = (row.get('true_level') or '').strip()
        if not sid or not marker or not level:
            skipped += 1
            continue
        try:
            db.save_pad_labels(sid, [{
                'marker':        marker,
                'pad_index':     int(row.get('pad_index', 0)),
                'true_level':    level,
                'measured_value':row.get('measured_value'),
                'label_source':  row.get('label_source', 'csv_import'),
                'pad_color':     row.get('pad_color'),
                'white_ref_rgb': row.get('white_ref_rgb'),
            }])
            saved += 1
        except Exception as exc:
            errors.append({'row': i + 2, 'error': str(exc)})

    return jsonify({'saved': saved, 'skipped': skipped, 'errors': errors, 'success': True})


# ---------------------------------------------------------------------------
# Roboflow YOLOv8 Dataset ZIP Import
# ---------------------------------------------------------------------------

@dataset_bp.route('/api/datasets/<dataset_id>/import/yolo_zip', methods=['POST'])
def import_yolo_zip(dataset_id):
    """
    Import a Roboflow YOLOv8 export ZIP containing:
    - data.yaml
    - train/images/, train/labels/
    - valid/images/, valid/labels/
    - (optional) test/images/, test/labels/
    """
    db = _db()
    ds = db.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Only .zip files are supported'}), 400

    target_dir = os.path.join(current_app.config.get('UPLOAD_DIR', 'uploads'), 'yolo_datasets', dataset_id)
    os.makedirs(target_dir, exist_ok=True)

    imported_images = 0
    imported_labels = 0
    splits_found = []

    try:
        with zipfile.ZipFile(f) as zf:
            for member in zf.namelist():
                # Prevent directory traversal attacks
                filename = os.path.basename(member)
                if not filename:
                    continue

                lower_mem = member.lower()
                # Determine split
                split = 'train' if 'train/' in lower_mem or 'train\\' in lower_mem else (
                    'valid' if 'valid/' in lower_mem or 'val/' in lower_mem or 'valid\\' in lower_mem or 'val\\' in lower_mem else (
                        'test' if 'test/' in lower_mem or 'test\\' in lower_mem else None
                    )
                )

                if split and split not in splits_found:
                    splits_found.append(split)

                # Extract and preserve Roboflow structure
                # member paths look like: 'train/images/foo.jpg', 'valid/labels/bar.txt', 'data.yaml'
                norm_mem = member.replace('\\', '/')
                dest_path = os.path.join(target_dir, norm_mem)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                with open(dest_path, 'wb') as out_f:
                    out_f.write(zf.read(member))

                # If it's an image, also register as a sample in the dataset database
                if any(filename.lower().endswith(f'.{ext}') for ext in ALLOWED_EXTENSIONS):
                    sample_id = db.add_sample(
                        dataset_id=dataset_id,
                        image_path=dest_path,
                        strip_type=ds.get('strip_type', 'URS-4K'),
                        capture_meta={'original_filename': filename, 'yolo_split': split or 'train'},
                        split=split or 'train',
                        is_synthetic=0
                    )
                    imported_images += 1

                elif filename.lower().endswith('.txt') and filename != 'data.yaml':
                    # Clean segmentation / polygon rows into standard bbox if needed
                    try:
                        with open(dest_path, 'r') as lf:
                            lines = lf.readlines()
                        cleaned_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cleaned_lines.append(line.strip())
                            elif len(parts) > 5:
                                cls_id = parts[0]
                                coords = [float(x) for x in parts[1:]]
                                xs = coords[0::2]
                                ys = coords[1::2]
                                xmin, xmax = min(xs), max(xs)
                                ymin, ymax = min(ys), max(ys)
                                cx = (xmin + xmax) / 2.0
                                cy = (ymin + ymax) / 2.0
                                w = xmax - xmin
                                h = ymax - ymin
                                cleaned_lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
                        with open(dest_path, 'w') as lf:
                            lf.write('\n'.join(cleaned_lines) + '\n')
                    except Exception:
                        pass
                    imported_labels += 1

    except zipfile.BadZipFile:
        return jsonify({'error': 'Corrupted or invalid ZIP file'}), 400
    except Exception as e:
        logger.error(f"Error extracting YOLO zip: {e}")
        return jsonify({'error': str(e)}), 500

    # Ensure data.yaml points to absolute paths for YOLO training
    yaml_path = os.path.join(target_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as yf:
                content = yf.read()
            train_p = os.path.join(target_dir, 'train', 'images').replace('\\', '/')
            val_p   = os.path.join(target_dir, 'valid', 'images').replace('\\', '/')
            test_p  = os.path.join(target_dir, 'test', 'images').replace('\\', '/')
            new_lines = []
            for line in content.splitlines():
                if line.strip().startswith('train:'):
                    new_lines.append(f'train: {train_p}')
                elif line.strip().startswith('val:'):
                    new_lines.append(f'val: {val_p}')
                elif line.strip().startswith('test:'):
                    new_lines.append(f'test: {test_p}')
                else:
                    new_lines.append(line)
            with open(yaml_path, 'w') as yf:
                yf.write('\n'.join(new_lines) + '\n')
        except Exception as ye:
            logger.warning(f"Could not rewrite data.yaml paths: {ye}")

    return jsonify({
        'success': True,
        'dataset_id': dataset_id,
        'imported_images': imported_images,
        'imported_labels': imported_labels,
        'splits': splits_found,
        'storage_path': target_dir
    }), 201
