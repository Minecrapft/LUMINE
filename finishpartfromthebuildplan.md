# LUMINE Build Plan — Completed Parts Tracker
_Last updated: 2026-10-28_  
🧪 **Test Suite Status: 102 passed** — `tests/test_analysis.py` includes the saliva pH workflow

---

## ✅ Phase 0 — Audit & Verification (M0)

**Status:** COMPLETE — Tests Green ✅

### What was done
- Read every cited source file and verified all 5 original audit claims.
- Determined exact pad counts, marker lists, and chart availability.
- Checked deployment mode (Flask dev server, single-worker), upload location (`uploads/`), and auth model (none — noted as open question for Phase 6).
- Confirmed `tests/test_analysis.py`, `test_camera.py`, `test_health.py` were 0 bytes.
- Created `docs/audit.md` with confirmed/denied findings and answers to all "Also determine" questions.
- Created `tests/conftest.py` (pytest sys.path setup).
- Created `tests/test_analysis.py` with Phase 0 smoke tests + Phase 1 acceptance tests.

### Files created / modified
| File | Action |
|---|---|
| `docs/audit.md` | **Created** — full verified audit report |
| `tests/conftest.py` | **Created** — pytest sys.path setup |
| `tests/test_analysis.py` | **Created** — Phase 0 smoke + Phase 1 acceptance tests |

---

## ✅ Phase 1 — Color References + Lab Baseline (M1, part 1 of 3)

**Status:** COMPLETE

### What was done
1. **`app/ml/color.py` (new):** sRGB → CIE L*a*b* (D65) via `scikit-image.color.rgb2lab` + CIEDE2000 `deltaE_ciede2000`. Includes pure-Python fallback if skimage absent.
2. **`app/ml/baseline.py` (new):** `predict_level(measured_rgb, reference_entries)` — nearest-reference predictor using ΔE00. Returns `level`, `delta_e_best`, `delta_e_second`, `confidence`, `low_confidence`. Confidence derived from margin between best and second-best match.
3. **`app/ml/chart_colors.py` (updated):** Added `get_chart(profile_id)` accessor; fixed `PROFILE_CHARTS` keys to `'URS-4K'` / `'SALIVA-4K'` matching DB profile IDs.
4. **`app/ml/__init__.py` (updated):** Removed broken `PadClassifier` import (file never existed). Exposed `rgb_to_lab`, `delta_e`, `predict_level`, chart accessors.
5. **`app/database/db_manager.py` (updated):**
   - Added import of `PROFILE_CHARTS` from `chart_colors`.
   - Replaced the old hardcoded SALIVA-4K grayscale seed block with a new `_seed_reference_colors_from_chart(cursor)` method that idempotently inserts reference colors for **both** `URS-4K` and `SALIVA-4K` from the manufacturer-approximated chart.
   - Added public `seed_reference_colors_idempotent()` method.
   - **URS-4K now has reference colors** (was completely missing before).
   - **SALIVA-4K now uses chart-based perceptual colors** (replaced the nearly-all-gray entries).
6. **`app/routes.py` (updated):**
   - Added import of `predict_level` from `app.ml.baseline`.
   - **Removed the hardcoded grayscale fallback** (`{'level': 'negative', 'rgb': [240, 240, 240]}` etc.).
   - `/api/analyze` now calls `predict_level()` per marker; returns `no_reference` error entry if no refs found instead of silent fallback.
   - Response now matches the plan's contract: `status`, `method`, `quality`, `detection`, `results[]` with `delta_e_best`/`delta_e_second`/`low_confidence`, `model`, `disclaimer`.
   - Kept all legacy response fields (`success`, `markers`, `summary`, etc.) for backward-compat with existing frontend.
7. **`requirements.txt` (updated):** Added `scikit-image`, `joblib`. Removed `ultralytics`, `torch`, `torchvision`, `colour-science`, `picamera2` (out of scope per plan). Changed `opencv-python` → `opencv-python-headless`.

### Files created / modified
| File | Action |
|---|---|
| `app/ml/color.py` | **Created** — CIE Lab conversion + CIEDE2000 |
| `app/ml/baseline.py` | **Created** — ΔE00 nearest-reference predictor |
| `app/ml/chart_colors.py` | **Updated** — added `get_chart()`, fixed profile key names |
| `app/ml/__init__.py` | **Updated** — correct exports, removed broken import |
| `app/database/db_manager.py` | **Updated** — chart-based color seeding for URS-4K + SALIVA-4K |
| `app/routes.py` | **Updated** — Lab baseline in `/api/analyze`, removed grayscale fallback, new response contract |
| `requirements.txt` | **Updated** — correct dependency set per plan Section 6 |

---

## ✅ Phase 2 — Strip & Pad Localization (Computer Vision)

**Status:** COMPLETE — 12 New Tests Green ✅

### What was done
1. **`app/image_processing/strip_detection.py` (rewritten):**
   - Replaced fixed center crop with real multi-cue contour detection (morphological gradient, Canny edges with closure, adaptive thresholding, saturation maps, Otsu).
   - Geometric candidate scoring using aspect ratio fit, rectangularity, solidity, and area ratio.
   - Rotated bounding rectangle (`cv2.minAreaRect`) and perspective rectification (`cv2.warpPerspective`) to produce an axis-aligned strip.
   - Canonical horizontal orientation normalization (automatically orienting the pad cluster to the left and blank handle to the right via variance analysis).
   - `StripNotFound` exception raised when detection confidence is below threshold or no plausible strip exists.
   - Manual 4-corner override fallback (`manual_corners`) for UI interaction.
2. **`app/image_processing/pad_detection.py` (rewritten):**
   - Prior-driven pad pitch and position estimation along the active zone of the rectified strip.
   - 1D Sobel gradient profile refinement along the strip axis to snap to pad centers.
   - Inner region sampling (central 55% of pad width/height) to eliminate substrate bleed, border shadows, and edge meniscus artifacts.
   - Per-pad color extraction (mean RGB + normalized chromaticity).
   - Per-pad dispersion (mean pixel standard deviation across channels) and `high_variance` flag for detecting glare or blotchiness.
   - Backward-mapping of pad bounding boxes and centers to original image coordinates via inverse perspective matrix (`inv_transform_matrix`).
   - Debug overlay generator (`generate_debug_overlay`): renders green strip boundary quad, cyan pad bounding boxes, pad indices (`#0`–`#9`), center dots, and confidence banner.
3. **`app/image_processing/__init__.py` (updated):**
   - Cleanly exports `StripDetector`, `StripNotFound`, `StripNotFoundError`, `PadDetector`, `ColorAnalyzer`.
4. **`app/routes.py` (updated):**
   - Accepts optional `manual_corners` in `/api/analyze`.
   - Catches `StripNotFound` and returns contract-compliant `status: "retake"` with descriptive reason and retake prompt instead of guessing.
   - Generates and writes debug overlay image (`overlay_<filename>.jpg`) and populates `detection.overlay_image` in the API response.
5. **`tests/test_analysis.py` (updated):**
   - Added 12 acceptance tests covering 0°, 30°, 90° rotations on both dark and light backgrounds, blank image rejection, manual corner override, 10-pad localization accuracy (< 15px error), dispersion flags, debug overlay generation, and `/api/analyze` retake response.

### Files created / modified
| File | Action |
|---|---|
| `app/image_processing/strip_detection.py` | **Rewritten** — Multi-cue contour detection, `warpPerspective`, `StripNotFound`, manual corners |
| `app/image_processing/pad_detection.py` | **Rewritten** — Prior-driven pad detection, 1D profile refinement, inner crop, dispersion, debug overlay |
| `app/image_processing/__init__.py` | **Updated** — Exported image processing symbols |
| `app/routes.py` | **Updated** — StripNotFound retake handling, manual corners support, debug overlay file saving |
| `tests/test_analysis.py` | **Updated** — 12 new Phase 2 unit & integration tests (35 total passing) |

---

## ✅ Phase 3 — White Reference, Quality Gates & Analyze Integration

**Status:** COMPLETE — 7 New Tests Green ✅ (42 total)

### What was done
1. **`app/image_processing/white_reference.py` (new):**
   - `WhiteReferenceNormalizer.normalize_strip_pads(rectified_strip, pads)`: samples the plastic substrate (handle region, rightmost 15% of rectified strip) to compute a per-channel mean white RGB.
   - Computes per-channel gain = `target_white / substrate_white` (clamped 0.5–3.0) to correct for warm/cool ambient illumination before CIE L*a*b* conversion.
   - Handles edge cases: all-black substrate, single-pad lists, and already-normalized pads.
2. **`app/image_processing/quality_gates.py` (new):**
   - `QualityGateChecker.evaluate_image(image)`: runs blur gate (Laplacian variance < 50 → retake), exposure gate (mean brightness < 30 or > 225 → retake), and glare gate (>5% saturated pixels → retake).
   - `QualityGateChecker.evaluate_strip_resolution(strip_result)`: rejects strips that are too small to reliably segment pads (< 100 px along dominant axis).
   - All gates return `{"pass": bool, "reason": str, "value": float}` dicts.
3. **`app/image_processing/__init__.py` (updated):** Exported `WhiteReferenceNormalizer` and `QualityGateChecker`.
4. **`app/routes.py` (updated):**
   - `/api/analyze` now runs quality gates before strip detection; returns `status: "retake"` with human-readable `reason` if any gate fails.
   - White reference normalization applied to all pad colors before Lab baseline prediction.
   - Debug overlay file (`overlay_<filename>.jpg`) generated and path included in response.
   - Full Section 5 response contract implemented: `status`, `method`, `quality`, `detection`, `results[]`, `model`, `disclaimer`.
5. **`tests/test_analysis.py` (updated):** Added 7 Phase 3 acceptance tests covering blur gate, exposure gate (dark & overexposed), glare gate, resolution gate, white reference normalization, and end-to-end synthetic strip → expected levels pipeline.

### Files created / modified
| File | Action |
|---|---|
| `app/image_processing/white_reference.py` | **Created** — substrate white sampling, per-channel gain normalization |
| `app/image_processing/quality_gates.py` | **Created** — blur, exposure, glare, and resolution gates |
| `app/image_processing/__init__.py` | **Updated** — exported `WhiteReferenceNormalizer`, `QualityGateChecker` |
| `app/routes.py` | **Updated** — quality gates wired in, retake responses, white reference, full response contract |
| `tests/test_analysis.py` | **Updated** — 7 new Phase 3 tests (42 total passing) |

---

## ✅ Phase 4 — Dataset Management & Annotation

**Status:** COMPLETE (2026-10-28)

### What was done
1. **Database tables & CRUD (`app/database/db_manager.py`):**
   - Added tables: `training_datasets`, `training_samples`, `training_pad_labels`.
   - Methods: `create_dataset`, `get_dataset`, `list_datasets`, `delete_dataset` (cascading), `add_sample`, `get_sample`, `list_samples`, `delete_sample`, `save_pad_labels` (upsert per sample/marker/pad_index), `get_pad_labels`, `export_dataset_csv`, `assign_group_split`.
   - Group split ensures photos are cleanly partitioned into `train`, `val`, and `test` without photo leakage across splits.
2. **Phase 4 REST API (`app/routes_dataset.py`):**
   - Registered `dataset_bp` blueprint in `app.py`.
   - REST endpoints for dataset management, sample image upload (sanitized filenames, type validation), pad labeling, group partitioning, CSV export, and full ZIP export (CSV + images).
3. **Annotation Studio (`app/templates/annotation.html`):**
   - UI for dataset selection, sample viewing, pad level annotation with source tracking (`chart`, `standard_solution`, `visual`), optional numeric measured values, annotator logging, group splitting, and export triggers.
4. **Synthetic strip generator (`scripts/make_synthetic.py`):**
   - CLI generator drawing realistic multi-pad strips with colors sampled from `chart_colors.py`, lighting gradients, noise, and explicit synthetic watermarking.
   - Automatic DB ingestion flag `--add-to-db` tagging `is_synthetic=1`.
5. **Phase 4 tests (`tests/test_analysis.py`):**
   - 16 new tests (`TestPhase4DatasetDB` and `TestPhase4SyntheticGenerator`) covering DB CRUD, label upserts, cascade deletes, group split integrity, synthetic generation for URS-4K and SALIVA-4K, chart level validation, and DB ingestion.
   - All 58/58 tests passing.

### Files created / modified
| File | Action |
|---|---|
| `app/routes_dataset.py` | **Created** — REST endpoints for dataset & sample CRUD, export, split |
| `app/templates/annotation.html` | **Created** — Annotation studio page |
| `scripts/make_synthetic.py` | **Created** — Synthetic test strip generator |
| `app/database/db_manager.py` | **Updated** — Dataset schema and queries |
| `app.py` | **Updated** — Registered `dataset_bp` |
| `tests/test_analysis.py` | **Updated** — 16 Phase 4 unit tests (58 total passing) |

---

## ✅ Phase 5 — Training Engine

**Status:** COMPLETE (2026-10-28)

### What was done
1. **Feature extraction (`app/ml/features.py`):**
   - 24-dimensional feature vector per pad: Lab mean/median, HSV mean/median, normalized chromaticity r/g, dispersion, and white-reference differences (ΔL, Δa, Δb, ΔECIEDE2000, per-channel ratios).
   - `FEATURE_VERSION = 1` embedded in all artifacts for compatibility checking.
   - `extract_pad_features(patch_or_rgb, white_ref_rgb)` and `extract_features_batch()` APIs.
2. **Training engine (`app/ml/trainer.py`):**
   - `train()` function: loads DB data, validates minimum samples per class, group-aware train/val/test split, fits RF or GB classifier, evaluates accuracy/macro-F1/MALE/confusion matrix, saves joblib artifact with SHA-256 hash.
   - `load_model(path, sha256)`: loads and verifies by hash — raises `ValueError` on mismatch.
   - `predict(artifact, feat)`: returns `(level, confidence)` tuple.
   - Subprocess entry point (`python -m app.ml.trainer --job-id ...`) for background training.
3. **Model registry + training jobs (`app/database/db_manager.py`):**
   - Tables: `ml_models` (versioned, hashed), `training_jobs` (status/logs).
   - Methods: `register_model`, `get_model`, `list_models`, `get_active_model`, `activate_model` (deactivates others), `deactivate_model`, `delete_model`, `create_training_job`, `update_training_job`, `get_training_job`, `list_training_jobs`.
4. **Training REST API (`app/routes_training.py`):**
   - `POST /api/training/jobs` — launches subprocess, returns job_id.
   - `GET  /api/training/jobs` / `GET /api/training/jobs/<id>` — poll status.
   - `GET  /api/training/models` (filtered), `GET /api/training/models/<id>`.
   - `POST /api/training/models/<id>/activate` — with synthetic-only guard.
   - `POST /api/training/models/<id>/deactivate` — rollback.
   - `DELETE /api/training/models/<id>` — refuses active models.
   - Registered in `app.py` as `training_bp`.
5. **Phase 5 tests (18 new tests):**
   - `TestPhase5Features`: shape, white-diff values, ΔECIEDE2000, feature version constant, batch extraction.
   - `TestPhase5ModelRegistry`: register, version increment, activate/deactivate mutual exclusion, hash verification, delete.
   - `TestPhase5TrainingJobs`: create, poll, complete (model_id recorded), fail (error recorded).
   - `TestPhase5TrainEndToEnd`: full fit→save→load→predict cycle; refuses training on too few samples; group split has no overlap.
   - **76/76 tests passing (11.39s)**.

### Files created / modified
| File | Action |
|---|---|
| `app/ml/features.py` | **Created** — 24-dim feature extractor, versioned |
| `app/ml/trainer.py` | **Created** — RF/GB training pipeline, subprocess entry point, hash-verified model load |
| `app/routes_training.py` | **Created** — REST API for training jobs and model registry |
| `app/database/db_manager.py` | **Updated** — ml_models, training_jobs tables + methods; pad_color, white_ref_rgb in labels |
| `app.py` | **Updated** — Registered `training_bp` |
| `tests/test_analysis.py` | **Updated** — 18 Phase 5 tests (76 total passing) |

---

## 🔲 Phase 6 — ML Studio UI + Activation

## ✅ Phase 6 — ML Studio UI + Activation

**Status:** COMPLETE (2026-10-28)

### What was done
1. **ML model inference wired into `/api/analyze` (`app/routes.py`):**
   - For each pad marker, first checks `db.get_active_model(strip_type, marker)`.
   - If an active model exists, loads it (SHA-256 verified, cached in `_ml_model_cache`), extracts the 24-dim feature vector, and calls `predict()`.
   - Falls back to the Lab baseline if no active model, load fails, or feature version mismatches.
   - Response `method` field is now `"ml"` or `"baseline"` per call; `model.id` and `model.version` are populated.
2. **`/ml_studio` page route (`app/routes.py`):**
   - Passes `datasets`, `models`, and `jobs` to the template.
3. **`app/templates/ml_studio.html` — Admin ML Studio:**
   - **Datasets tab:** create dataset (name/strip/description), dataset library table with sample/label counts, Annotate / Export CSV / Delete buttons per dataset.
   - **Training tab:** launch form (strip, marker, algorithm, estimators, min_per_class, dataset checkboxes), live progress bar with 2-second polling, auto-switches to Models tab on completion.
   - **Models tab:** filter by marker/strip/active, table with val accuracy/F1/MALE per model, Activate button (with synthetic-only guard modal + override checkbox), Deactivate (rollback), Delete (blocked for active models), Metrics modal showing confusion matrix, top-10 feature importance, and split-level metrics.
   - **Jobs tab:** recent jobs table with refresh, status badges, mini progress bars.
4. **Nav link added to `base.html`:** "ML Studio" appears in the main navigation.

### Files created / modified
| File | Action |
|---|---|
| `app/routes.py` | **Updated** — ML model inference in `/api/analyze`; added `/ml_studio` page route |
| `app/templates/ml_studio.html` | **Created** — Full admin ML Studio (datasets, training, models, jobs) |
| `app/templates/base.html` | **Updated** — ML Studio nav link added |

---

## ✅ Phase 7 — Hardening & Documentation

**Status:** COMPLETE (2026-10-28)

### What was done
1. **Syntax / lint pass:** `py_compile` clean on all new modules (`routes.py`, `routes_dataset.py`, `routes_training.py`, `ml/features.py`, `ml/trainer.py`, `database/db_manager.py`, `scripts/make_synthetic.py` and all others).
2. **Config hardening (`app.py`, `config.py`):** Added `DATABASE` and `MODELS_DIR` config keys so the training blueprint always resolves the right paths regardless of working directory.
3. **Flask smoke test:** All key routes confirmed HTTP 200 with a live `test_client()` call:
   - `/ml_studio` → 200
   - `/api/training/models` → 200
   - `/api/datasets` → 200
   - `/api/training/jobs` → 200
4. **README.md (new file):** Comprehensive project documentation covering:
   - Architecture overview with module tree
   - Quick-start installation and environment variables
   - **Photo capture protocol** — camera distance (20–30 cm), lighting (diffuse, consistent, same each session), glare prevention (tilt ≤10°, diffuser sheet), strip reading window, white reference guidance
   - **ML training workflow** — collect real samples, generate synthetics, train, evaluate, activate / rollback
   - Full API reference table (analysis, dataset, training endpoints)
   - **Known limitations and validation status** — honest table of what is vs. is not verified; recommended 5-step validation protocol before real use; security notes
   - Directory layout
5. **Final test run:** 76/76 tests passing in 20.16 s.

### Files created / modified
| File | Action |
|---|---|
| `README.md` | **Created** — full project documentation |
| `app.py` | **Updated** — `DATABASE` and `MODELS_DIR` config defaults |
| `config.py` | **Updated** — `MODELS_DIR` class attribute |

---

## ✅ Phase 8 — YOLOv8 Integration & Pi 5 Optimization (M6)

**Status:** COMPLETE (2026-10-29)

### What was done
1. **Roboflow YOLOv8 Dataset Importer (`app/routes_dataset.py`):**
   - Added `POST /api/datasets/<dataset_id>/import/yolo_zip` endpoint.
   - Extracts complete Roboflow YOLOv8 export ZIPs (`data.yaml`, `train/`, `valid/`, `test/`).
   - Automatically registers all imported photos and coordinate files into the LUMINE database.
2. **ML Studio UI Update (`app/templates/ml_studio.html`):**
   - Added **"Roboflow YOLOv8 (.zip)"** tab under Dataset Upload Panel.
   - Allows 1-click drag-and-drop upload of the Roboflow `.zip`.
   - Added `YOLOv8-Nano (Strip Detector / ONNX)` choice in Training tab algorithm selector.
3. **Raspberry Pi 5 Inference Engine (`app/image_processing/yolo_detector.py`):**
   - Created `ONNXYOLODetector` powered by OpenCV DNN (`cv2.dnn.readNetFromONNX`).
   - **Zero PyTorch/Ultralytics runtime requirement on Raspberry Pi 5.**
   - CPU-optimized inference with automatic NMS filtering and bounding box mapping.

## ✅ Phase 9 — Four-Pad Saliva pH Demo

**Status:** IMPLEMENTED — Research use only

### What was done
1. Added a separate `SALIVA-PH4` profile with four physical pads and one combined pH result. The legacy `SALIVA-4K` profile remains unchanged.
2. Added an image-derived pH 0–14 reference chart and joint four-color matching that tolerates either strip orientation.
3. Enabled Saliva in sample selection and added one-photo capture guidance requiring all four pads in frame.
4. Results show one pH estimate and four measured/reference swatch pairs. The workflow is explicitly research-only and does not claim clinical validity.
5. Added profile, orientation, analysis, and results-rendering tests.
6. Left reaction time unconfigured; users should follow the strip packaging instructions.

### Known limitations
- Reference colors were sampled from supplied product images; they are approximate, uncalibrated, and not manufacturer-verified.
- Camera, lighting, and image processing can shift color matches. The result is a demo estimate, not a medical test.

---

## Open Questions for User (from build plan Section 8)

| # | Question | Status |
|---|---|---|
| 1 | Real number of pads/markers for URS-4K and SALIVA-4K? Colors in chart_colors.py match those strips? | **Answered** — 10 pads each; chart colors are manufacturer-approximated (not from specific brand datasheet). |
| 2 | Real labeled strip photos available? How many, with which phone/lighting? | **Pending** — needed for Phases 4–6. |
| 3 | Is there user login/roles? How should /ml_studio be protected? | **Pending** — no login system currently; needs decision before Phase 6. |
| 4 | Deployment target: laptop, Raspberry Pi, or hosted server? | **Partially answered** — Flask dev server, likely laptop for now. |
| 5 | Intended use: research/demo or real patients? | **Pending** — affects disclaimer requirements. |
