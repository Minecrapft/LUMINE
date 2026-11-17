"""
LUMINE — Phase 5 Training Engine Routes

Provides REST API for:
- Launching training jobs (subprocess)
- Polling job status and logs
- Listing and managing trained models
- Activating / deactivating / rolling back models
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from app.database.db_manager import TrainingCapacityError

logger = logging.getLogger(__name__)

training_bp = Blueprint("training", __name__, url_prefix="/api/training")


def _db():
    from app.database.db_manager import DatabaseManager
    db_path = current_app.config.get("DATABASE", "lumine.db")
    return DatabaseManager(db_path)


# In-memory dictionary tracking running job processes: {job_id: subprocess.Popen}
_RUNNING_PROCESSES = {}
_PROCESS_LOCK = threading.Lock()
_ACTIVE_STATUSES = {"pending", "running", "paused"}
_STALE_JOB_SECONDS = 60


def _process_matches_job(process_id, job_id, algorithm):
    """Confirm a persisted PID still belongs to this training job."""
    if not process_id:
        return False

    with _PROCESS_LOCK:
        process = _RUNNING_PROCESSES.get(job_id)
    if process is not None:
        return process.poll() is None

    if sys.platform.startswith("linux"):
        try:
            command = Path(f"/proc/{int(process_id)}/cmdline").read_bytes()
        except (OSError, ValueError):
            return False

        args = command.decode(errors="replace").split("\0")
        module = "app.ml.yolo_trainer" if algorithm == "yolov8_nano" else "app.ml.trainer"
        return (
            "-m" in args
            and module in args
            and any(
                args[index] == "--job-id" and index + 1 < len(args)
                and args[index + 1] == job_id
                for index in range(len(args))
            )
        )

    try:
        os.kill(int(process_id), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _process_group_exists(process_id):
    try:
        os.killpg(int(process_id), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_worker_process(process_id, process=None):
    """Stop a worker and its child processes, escalating after a short grace period."""
    process_id = int(process_id)
    if os.name == "posix":
        try:
            os.killpg(process_id, signal.SIGTERM)
        except ProcessLookupError:
            return

        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and _process_group_exists(process_id):
            time.sleep(0.1)
        if _process_group_exists(process_id):
            os.killpg(process_id, signal.SIGKILL)
        if process is not None:
            process.wait(timeout=2)
        return

    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        return

    os.kill(process_id, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(process_id, 0)
        except OSError:
            return
        time.sleep(0.1)
    os.kill(process_id, getattr(signal, "SIGKILL", signal.SIGTERM))


def _mark_job_failed_if_active(db, job_id, message):
    job = db.get_training_job(job_id)
    if job and job["status"] in _ACTIVE_STATUSES:
        db.update_training_job(
            job_id,
            status="failed",
            error_message=message,
            log_append=f"[ERROR] {message}\n",
        )


def _reconcile_job(db, job):
    """Mark a job failed when its worker exited without recording completion."""
    if not job or job["status"] not in _ACTIVE_STATUSES:
        return job

    process_id = job.get("process_id")
    if not process_id:
        try:
            updated = datetime.fromisoformat(job["updated_at"])
            age_seconds = (datetime.utcnow() - updated).total_seconds()
        except (TypeError, ValueError):
            age_seconds = _STALE_JOB_SECONDS + 1
        if age_seconds > _STALE_JOB_SECONDS:
            _mark_job_failed_if_active(
                db, job["job_id"], "Training worker exited before its process ID was recorded."
            )
            return db.get_training_job(job["job_id"])
        return job

    if not _process_matches_job(process_id, job["job_id"], job["algorithm"]):
        _mark_job_failed_if_active(
            db, job["job_id"], "Training worker exited without recording a final job status."
        )
        return db.get_training_job(job["job_id"])
    return job


def _reconcile_active_jobs(db):
    for job in db.list_active_training_jobs():
        _reconcile_job(db, job)


def _capture_process_output(job_id, process, db):
    pending_lines = []
    last_flush = time.monotonic()
    try:
        if process.stdout is not None:
            for line in process.stdout:
                pending_lines.append(line)
                now = time.monotonic()
                if len(pending_lines) >= 20 or now - last_flush >= 1:
                    db.update_training_job(job_id, log_append="".join(pending_lines))
                    pending_lines.clear()
                    last_flush = now

        if pending_lines:
            db.update_training_job(job_id, log_append="".join(pending_lines))
    except Exception:
        logger.exception("Could not read training output for job %s", job_id)


def _monitor_process(job_id, process, db):
    """Drain worker output and record unexpected process exits."""
    output_thread = threading.Thread(
        target=_capture_process_output,
        args=(job_id, process, db),
        name=f"training-output-{job_id}",
        daemon=True,
    )
    output_thread.start()
    try:
        exit_code = process.wait()
        output_thread.join(timeout=2)
        job = db.get_training_job(job_id)
        if job and job["status"] in _ACTIVE_STATUSES:
            message = (
                f"Training worker exited with code {exit_code} "
                "without recording a final job status."
            )
            _mark_job_failed_if_active(db, job_id, message)
    except Exception:
        logger.exception("Could not monitor training process for job %s", job_id)
    finally:
        with _PROCESS_LOCK:
            if _RUNNING_PROCESSES.get(job_id) is process:
                _RUNNING_PROCESSES.pop(job_id, None)


# ── Training Jobs ─────────────────────────────────────────────────────────────

@training_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """List recent training jobs."""
    db = _db()
    _reconcile_active_jobs(db)
    jobs = db.list_training_jobs(limit=50)
    return jsonify({"jobs": jobs})


@training_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    """Poll status and log output for a training job."""
    db = _db()
    job = _reconcile_job(db, db.get_training_job(job_id))
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    return jsonify(job)


@training_bp.route("/jobs/<job_id>/pause", methods=["POST"])
def pause_job(job_id):
    """Pause a YOLO training run at the next batch boundary."""
    db = _db()
    job = db.get_training_job(job_id)
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    if job["algorithm"] != "yolov8_nano":
        return jsonify({"error": "Pause is only supported for YOLOv8 Nano jobs"}), 409
    if job["status"] != "running":
        return jsonify({"error": f"Cannot pause a job with status '{job['status']}'"}), 409

    db.update_training_job(
        job_id,
        status="paused",
        log_append="[PAUSED] Training will pause after the current batch.\n",
    )
    return jsonify({"ok": True, "job_id": job_id, "status": "paused"})


@training_bp.route("/jobs/<job_id>/resume", methods=["POST"])
def resume_job(job_id):
    """Resume a YOLO training run paused at a batch boundary."""
    db = _db()
    job = db.get_training_job(job_id)
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    if job["algorithm"] != "yolov8_nano":
        return jsonify({"error": "Resume is only supported for YOLOv8 Nano jobs"}), 409
    if job["status"] != "paused":
        return jsonify({"error": f"Cannot resume a job with status '{job['status']}'"}), 409

    db.update_training_job(
        job_id,
        status="running",
        log_append="[RESUMED] Training resumed.\n",
    )
    return jsonify({"ok": True, "job_id": job_id, "status": "running"})


@training_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel / stop a running training job subprocess."""
    db = _db()
    job = _reconcile_job(db, db.get_training_job(job_id))
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    if job["status"] not in _ACTIVE_STATUSES:
        return jsonify({"error": f"Cannot stop a job with status '{job['status']}'"}), 409

    with _PROCESS_LOCK:
        proc = _RUNNING_PROCESSES.get(job_id)
    if proc:
        try:
            _stop_worker_process(proc.pid, proc)
        except Exception as e:
            logger.warning(f"Error terminating process for {job_id}: {e}")
            return jsonify({"error": f"Could not stop training worker: {e}"}), 500
        with _PROCESS_LOCK:
            if _RUNNING_PROCESSES.get(job_id) is proc:
                _RUNNING_PROCESSES.pop(job_id, None)
    elif job.get("process_id") and _process_matches_job(
        job["process_id"], job_id, job["algorithm"]
    ):
        try:
            _stop_worker_process(job["process_id"])
        except Exception as exc:
            logger.warning(f"Error terminating persisted process for {job_id}: {exc}")
            return jsonify({"error": f"Could not stop training worker: {exc}"}), 500

    current = db.get_training_job(job_id)
    if current and current["status"] in _ACTIVE_STATUSES:
        db.update_training_job(
            job_id,
            status="cancelled",
            log_append="[CANCELLED] Training stopped by user.\n"
        )
    logger.info(f"Training job {job_id} cancelled by user")
    return jsonify({"ok": True, "job_id": job_id, "status": "cancelled"})


@training_bp.route("/jobs", methods=["POST"])
def start_training():
    """
    Launch a training job in a subprocess.

    Body JSON:
    {
        "strip_type": "URS-4K",
        "marker": "glucose",
        "algorithm": "rf",
        "n_estimators": 200,
        "dataset_ids": ["DS_XXX", "DS_YYY"],
        "min_per_class": 5
    }
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    strip_type = data.get("strip_type", "")
    algorithm = data.get("algorithm", "rf")
    marker = data.get("marker", "")
    dataset_ids = data.get("dataset_ids")
    try:
        epochs = int(data.get("epochs", 15))
        n_estimators = int(data.get("n_estimators", 200))
        min_per_class = int(data.get("min_per_class", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Training settings must be valid integers"}), 400

    if not isinstance(strip_type, str) or not isinstance(algorithm, str):
        return jsonify({"error": "strip_type and algorithm must be strings"}), 400
    strip_type = strip_type.strip()
    algorithm = algorithm.strip()
    if not isinstance(marker, str):
        return jsonify({"error": "marker must be a string"}), 400
    marker = marker.strip() if algorithm != "yolov8_nano" else "strip_detector"
    if not isinstance(dataset_ids, list) or not all(
        isinstance(dataset_id, str) and dataset_id.strip()
        for dataset_id in dataset_ids
    ):
        return jsonify({"error": "dataset_ids must be a list of dataset IDs"}), 400
    dataset_ids = list(dict.fromkeys(dataset_id.strip() for dataset_id in dataset_ids))

    if not strip_type:
        return jsonify({"error": "strip_type is required"}), 400
    if algorithm != "yolov8_nano" and not marker:
        return jsonify({"error": "marker is required for this algorithm"}), 400
    if not dataset_ids:
        return jsonify({"error": "dataset_ids must be a non-empty list"}), 400
    if algorithm not in ("rf", "gb", "yolov8_nano"):
        return jsonify({"error": "algorithm must be 'rf', 'gb', or 'yolov8_nano'"}), 400
    if not 1 <= epochs <= 300:
        return jsonify({"error": "epochs must be between 1 and 300"}), 400
    if not 10 <= n_estimators <= 2000:
        return jsonify({"error": "n_estimators must be between 10 and 2000"}), 400
    if not 1 <= min_per_class <= 100:
        return jsonify({"error": "min_per_class must be between 1 and 100"}), 400

    db = _db()
    _reconcile_active_jobs(db)

    if algorithm == "yolov8_nano":
        if len(dataset_ids) != 1:
            return jsonify({"error": "YOLO training requires exactly one selected dataset."}), 400
        dataset = db.get_dataset(dataset_ids[0])
        if not dataset:
            return jsonify({"error": f"Dataset {dataset_ids[0]} was not found."}), 404
        if dataset.get("strip_type") != strip_type:
            return jsonify({"error": "The selected dataset strip type must match the training strip type."}), 400
        upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")
        data_yaml = os.path.join(
            upload_dir, "yolo_datasets", dataset_ids[0], "data.yaml"
        )
        if not os.path.isfile(data_yaml):
            return jsonify({
                "error": "Import a Roboflow YOLO ZIP into the selected dataset before training."
            }), 400

    try:
        job_id = db.create_training_job(
            strip_type=strip_type,
            marker=marker,
            algorithm=algorithm,
            max_concurrent=1,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except TrainingCapacityError as exc:
        return jsonify({"error": str(exc)}), 409
    db.update_training_job(
        job_id, status="running", log_append=f"Job {job_id} started.\n"
    )

    db_path = current_app.config.get("DATABASE", "lumine.db")
    models_dir = current_app.config.get("MODELS_DIR", "models")

    # YOLOv8 training runs a dedicated subprocess (separate from RF/GB trainer)
    if algorithm == "yolov8_nano":
        upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")
        cmd = [
            sys.executable, "-m", "app.ml.yolo_trainer",
            "--job-id",     job_id,
            "--db",         db_path,
            "--strip-type", strip_type,
            "--epochs",     str(epochs),
            "--models-dir", models_dir,
            "--upload-dir", upload_dir,
            "--dataset-ids",
        ] + dataset_ids
    else:
        cmd = [
            sys.executable, "-m", "app.ml.trainer",
            "--job-id",       job_id,
            "--db",           db_path,
            "--strip-type",   strip_type,
            "--marker",       marker,
            "--algorithm",    algorithm,
            "--estimators",   str(n_estimators),
            "--models-dir",   models_dir,
            "--min-per-class",str(min_per_class),
            "--dataset-ids",
        ] + dataset_ids

    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        db.update_training_job(job_id, process_id=proc.pid)
        with _PROCESS_LOCK:
            _RUNNING_PROCESSES[job_id] = proc
        threading.Thread(
            target=_monitor_process,
            args=(job_id, proc, db),
            name=f"training-monitor-{job_id}",
            daemon=True,
        ).start()
        logger.info(f"Launched training subprocess for job {job_id}: {cmd}")
    except Exception as exc:
        if proc is not None:
            try:
                _stop_worker_process(proc.pid, proc)
            except Exception:
                logger.exception("Could not stop worker after launch setup failed")
            if proc.stdout is not None:
                proc.stdout.close()
        db.update_training_job(job_id, status="failed", error_message=str(exc))
        logger.error(f"Failed to launch training subprocess: {exc}")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"job_id": job_id, "status": "running"}), 202


# ── Model Registry ────────────────────────────────────────────────────────────

@training_bp.route("/models", methods=["GET"])
def list_models():
    """List all registered models, optionally filtered by strip_type/marker."""
    strip_type  = request.args.get("strip_type")
    marker      = request.args.get("marker")
    active_only = request.args.get("active_only", "false").lower() == "true"
    models = _db().list_models(strip_type=strip_type, marker=marker, active_only=active_only)
    return jsonify({"models": models})


@training_bp.route("/models/<model_id>", methods=["GET"])
def get_model(model_id):
    """Retrieve a single model's metadata and metrics."""
    model = _db().get_model(model_id)
    if not model:
        return jsonify({"error": f"Model {model_id} not found"}), 404
    return jsonify(model)


@training_bp.route("/models/<model_id>/activate", methods=["POST"])
def activate_model(model_id):
    """
    Activate a model for inference.

    Activation guard:
    - Refused if model has no recorded metrics.
    - Refused if all training data was synthetic (is_synthetic bypass requires
      explicit override flag in the request body).
    - Logs the activation event.
    """
    override = (request.get_json(force=True) or {}).get("override_synthetic_guard", False)

    db = _db()
    model = db.get_model(model_id)
    if not model:
        return jsonify({"error": f"Model {model_id} not found"}), 404

    metrics = model.get("metrics", {})
    if not metrics:
        return jsonify({
            "error": "Model has no recorded metrics. Cannot activate an untrained model."
        }), 422

    # Synthetic-only guard
    dataset_ids = model.get("dataset_ids", [])
    if dataset_ids:
        try:
            total = 0; synthetic = 0
            for ds_id in dataset_ids:
                samples = db.list_samples(ds_id)
                for s in samples:
                    total += 1
                    if s.get("is_synthetic"):
                        synthetic += 1
            if total > 0 and synthetic == total and not override:
                return jsonify({
                    "error": (
                        "All training samples are synthetic. "
                        "Activate with real data or pass override_synthetic_guard=true "
                        "with explicit acknowledgement."
                    ),
                    "synthetic_only": True
                }), 422
        except Exception:
            pass  # guard is best-effort

    ok = db.activate_model(model_id)
    if not ok:
        return jsonify({"error": "Activation failed"}), 500

    logger.info(f"Model {model_id} activated by request")
    return jsonify({
        "ok": True,
        "model_id": model_id,
        "strip_type": model["strip_type"],
        "marker": model["marker"],
        "version": model["version"],
        "algorithm": model["algorithm"],
        "activated_at": datetime.utcnow().isoformat()
    })


@training_bp.route("/models/<model_id>/deactivate", methods=["POST"])
def deactivate_model(model_id):
    """Deactivate (roll back) a model, reverting to baseline inference."""
    db = _db()
    ok = db.deactivate_model(model_id)
    if not ok:
        return jsonify({"error": f"Model {model_id} not found or already inactive"}), 404
    return jsonify({"ok": True, "model_id": model_id, "status": "deactivated"})


@training_bp.route("/models/<model_id>", methods=["DELETE"])
def delete_model_route(model_id):
    """Delete a model metadata record and its file on disk."""
    db = _db()
    model = db.get_model(model_id)
    if not model:
        return jsonify({"error": f"Model {model_id} not found"}), 404
    if model.get("is_active"):
        return jsonify({"error": "Cannot delete an active model. Deactivate it first."}), 422
    
    # Also delete the physical file on disk if it exists
    path = model.get("model_path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Could not remove model file {path}: {e}")

    db.delete_model(model_id)
    return jsonify({"ok": True, "deleted": model_id})
