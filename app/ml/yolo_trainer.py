"""
app/ml/yolo_trainer.py — YOLOv8-Nano training subprocess entry point for LUMINE.

Trains a YOLOv8-nano strip and pad detector from a Roboflow YOLO dataset imported into
the LUMINE database, then exports the best weights to ONNX for Raspberry Pi 5
inference via cv2.dnn.

Usage (subprocess entry point via routes_training.py):
    python -m app.ml.yolo_trainer \\
        --job-id JOB_XXX \\
        --db lumine.db \\
        --strip-type URS-4K \\
        --models-dir models/ \\
        --upload-dir uploads/ \\
        --dataset-ids DS_AAA DS_BBB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("lumine.yolo_trainer")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_job(db, job_id: str, status=None, progress=None, log_line=None, error=None, model_id=None):
    try:
        db.update_training_job(
            job_id,
            status=status,
            progress=progress,
            log_append=(log_line + "\n") if log_line else None,
            error_message=error,
            model_id=model_id,
        )
    except Exception as exc:
        logger.warning(f"Could not update job {job_id}: {exc}")


def _training_loss(trainer) -> float | None:
    """Return the latest aggregate YOLO training loss for progress reporting."""
    for attribute in ("tloss", "loss_items"):
        value = getattr(trainer, attribute, None)
        if value is None:
            continue

        try:
            components = value.values() if isinstance(value, Mapping) else (value,)
            loss_values = []
            for component in components:
                if hasattr(component, "detach"):
                    component = component.detach().cpu().reshape(-1).tolist()
                elif hasattr(component, "reshape") and hasattr(component, "tolist"):
                    component = component.reshape(-1).tolist()
                values = component if isinstance(component, (list, tuple)) else (component,)
                loss_values.extend(float(item) for item in values)

            loss = sum(loss_values)
            if loss_values and math.isfinite(loss):
                return loss
        except (TypeError, ValueError, RuntimeError):
            continue
    return None


def _wait_while_paused(db, job_id: str) -> None:
    """Keep the trainer process alive while a pause is requested."""
    import time

    pause_logged = False
    while True:
        job = db.get_training_job(job_id)
        if not job or job["status"] != "paused":
            if pause_logged and job and job["status"] == "running":
                _update_job(db, job_id, log_line="[RESUMED] Training continuing.\n")
            return
        if not pause_logged:
            _update_job(db, job_id, log_line="[PAUSED] Waiting to resume at batch boundary.\n")
            pause_logged = True
        time.sleep(1)


def _find_data_yaml(upload_dir: str, dataset_ids: list) -> str | None:
    """Return the YOLO config for the single selected imported dataset."""
    if len(dataset_ids) != 1:
        raise ValueError("YOLO training requires exactly one dataset.")
    candidate = os.path.join(upload_dir, "yolo_datasets", dataset_ids[0], "data.yaml")
    return candidate if os.path.isfile(candidate) else None


def train(
    job_id: str,
    db_path: str,
    strip_type: str,
    models_dir: str,
    upload_dir: str,
    dataset_ids: list,
    epochs: int = 15,
) -> None:
    # ── Import DB ──────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.database.db_manager import DatabaseManager
    db = DatabaseManager(db_path)

    def _log(msg: str, progress: int = None):
        logger.info(msg)
        _update_job(db, job_id, log_line=msg, progress=progress)

    _log("=== LUMINE YOLOv8-Nano Training ===")
    _log(f"Strip type : {strip_type}")
    _log(f"Dataset IDs: {dataset_ids}")

    # ── Locate data.yaml ───────────────────────────────────────────────────
    data_yaml = _find_data_yaml(upload_dir, dataset_ids)
    if not data_yaml:
        msg = "No data.yaml found. Import a Roboflow YOLOv8 ZIP first."
        _update_job(db, job_id, status="failed", error=msg, log_line=msg)
        logger.error(msg)
        return

    _log(f"Found data.yaml: {data_yaml}", progress=5)

    # ── Attempt YOLOv8 training via ultralytics ────────────────────────────
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        msg = ("ultralytics is not installed in this environment. "
               "Install it on your training machine with: pip install ultralytics")
        _update_job(db, job_id, status="failed", error=msg, log_line=msg)
        logger.error(msg)
        return

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        logger.debug("PyTorch inter-op thread count was already initialized")

    os.makedirs(models_dir, exist_ok=True)
    run_name = f"yolov8n_strip_{uuid.uuid4().hex[:8]}"

    _log("Initialising YOLOv8-Nano model...", progress=10)
    model = YOLO("yolov8n.pt")  # downloads nano weights on first run

    _log(f"Starting training (epochs={epochs}, imgsz=640)...", progress=15)
    
    # Callbacks to stream live progress into LUMINE DB & ML Studio UI
    import time
    total_epochs = max(1, epochs)
    last_batch_update = [time.time()]
    batch_count = [0]

    def on_train_epoch_start(trainer):
        batch_count[0] = 0
        curr = trainer.epoch + 1
        _update_job(db, job_id, log_line=f"--- Starting Epoch {curr}/{total_epochs} ---")

    def on_train_batch_end(trainer):
        batch_count[0] += 1
        _wait_while_paused(db, job_id)
        now = time.time()
        # Report progress every 8 seconds so the user sees live movement immediately
        if now - last_batch_update[0] >= 8.0:
            last_batch_update[0] = now
            nb = len(trainer.train_loader) if hasattr(trainer, 'train_loader') and trainer.train_loader else 1
            curr = trainer.epoch
            b_idx = min(batch_count[0], nb)
            frac = (curr + (b_idx / max(1, nb))) / total_epochs
            pct = min(80, max(15, int(15 + frac * 65)))
            loss = _training_loss(trainer)
            loss_info = f" | loss: {loss:.6f}" if loss is not None else ""
            _update_job(
                db,
                job_id,
                progress=pct,
                log_line=f"Epoch {curr + 1}/{total_epochs} [batch {b_idx}/{nb}]{loss_info}"
            )

    def on_train_epoch_end(trainer):
        curr_epoch = trainer.epoch + 1
        pct = min(80, int(15 + (curr_epoch / total_epochs) * 65))
        loss = _training_loss(trainer)
        loss_info = f" | loss: {loss:.6f}" if loss is not None else ""
        _update_job(
            db,
            job_id,
            progress=pct,
            log_line=f"✓ Epoch {curr_epoch}/{total_epochs} complete{loss_info}"
        )

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    try:
        results = model.train(
            data=data_yaml,
            epochs=total_epochs,
            imgsz=640,
            batch=4,
            workers=1,
            device="cpu",
            name=run_name,
            project=models_dir,
            verbose=False,
            exist_ok=True,
        )
    except Exception as train_exc:
        msg = f"Training failed: {train_exc}"
        logger.exception(msg)
        _update_job(db, job_id, status="failed", error=msg, log_line=msg)
        return

    best_pt = os.path.join(models_dir, run_name, "weights", "best.pt")
    if not os.path.exists(best_pt):
        msg = f"Training completed but best.pt not found at {best_pt}"
        _update_job(db, job_id, status="failed", error=msg, log_line=msg)
        return

    _log(f"Training complete. Best weights: {best_pt}", progress=80)

    # ── Export to ONNX (for cv2.dnn on Pi 5) ──────────────────────────────
    _log("Exporting to ONNX for Raspberry Pi 5 deployment...", progress=85)
    best_model = YOLO(best_pt)
    best_model.export(format="onnx", imgsz=640, simplify=True)

    onnx_path = best_pt.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        msg = f"ONNX export failed — {onnx_path} not found."
        _update_job(db, job_id, status="failed", error=msg, log_line=msg)
        return

    _log(f"ONNX model saved: {onnx_path}", progress=90)

    # ── Register in model registry ─────────────────────────────────────────
    sha = _sha256_file(onnx_path)
    model_names = best_model.names
    if isinstance(model_names, dict):
        class_names = [str(model_names[key]) for key in sorted(model_names)]
    else:
        class_names = [str(name) for name in model_names]
    metrics = {
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "onnx_path": onnx_path,
        "class_names": class_names,
    }

    model_id = db.register_model(
        strip_type=strip_type,
        marker="strip_detector",      # Registry key for the multi-class strip/pad detector
        algorithm="yolov8_nano",
        feature_version=0,            # N/A for detection models
        dataset_ids=dataset_ids,
        metrics=metrics,
        baseline_metrics={},
        model_path=onnx_path,
        sha256_hash=sha,
    )

    _update_job(db, job_id, status="completed", progress=100,
                log_line=f"Done. model_id={model_id}", model_id=model_id)
    _log(f"=== Training complete. model_id={model_id} ===", progress=100)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LUMINE YOLOv8-Nano training subprocess")
    parser.add_argument("--job-id",      required=True)
    parser.add_argument("--db",          required=True)
    parser.add_argument("--strip-type",  required=True)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--models-dir",  required=True)
    parser.add_argument("--upload-dir",  required=True)
    parser.add_argument("--dataset-ids", nargs="+", required=True)
    args = parser.parse_args()

    train(
        job_id=args.job_id,
        db_path=args.db,
        strip_type=args.strip_type,
        models_dir=args.models_dir,
        upload_dir=args.upload_dir,
        dataset_ids=args.dataset_ids,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
