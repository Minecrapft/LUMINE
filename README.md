# LUMINE — Colorimetric Strip Analysis Platform

> **Research use only.** LUMINE is a computer-vision and machine-learning research tool
> for analysing colorimetric reagent test strips. Results are **not** validated for
> clinical diagnosis and must **not** replace laboratory testing or professional medical advice.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Photo Capture Protocol](#photo-capture-protocol)
5. [ML Training Workflow](#ml-training-workflow)
6. [API Reference](#api-reference)
7. [Known Limitations and Validation Status](#known-limitations-and-validation-status)
8. [Directory Layout](#directory-layout)

---

## Overview

LUMINE is a Flask web application that:

- **Captures or uploads** a photo of a reagent test strip (URS-4K, SALIVA-4K, or similar).
- **Detects** the strip boundary and individual reagent pads using classical computer vision.
- **Normalises colour** against an on-strip white reference to reduce camera and lighting variation.
- **Classifies** each pad against a CIE Lab reference chart (baseline) or a trained ML model.
- **Interprets** results with a clinical marker engine and condition-association rules.
- **Stores** patients, assessments, training datasets, and trained models in a local SQLite database.
- **Exposes** an admin ML Studio for dataset management, model training, and activation.

---

## Architecture

```
app/
    routes.py               Main Flask routes + /api/analyze (baseline and ML inference)
    routes_dataset.py       Dataset management API  (/api/datasets/*)
    routes_training.py      Training job + model registry API  (/api/training/*)
    database/
        db_manager.py       SQLite persistence (patients, assessments, datasets, models, jobs)
    image_processing/
        strip_detection.py  Perspective-corrected strip localisation (MSER + homography)
        pad_detection.py    Uniform-grid pad cropping + colour sampling
        white_reference.py  On-strip white-patch normalisation
        quality_gates.py    Blur / exposure / glare / resolution gates
    ml/
        baseline.py         CIE Lab CIEDE2000 nearest-reference baseline predictor
        features.py         24-dim feature extractor (Lab + HSV + chromaticity + white-ref diff)
        trainer.py          RF / GB training pipeline, group-aware split, hash-verified artifacts
        chart_colors.py     Reference colour chart (RGB per strip / marker / level)
        color.py            sRGB <-> Lab conversions, CIEDE2000 delta-E
    health/
        marker_engine.py    Per-marker clinical interpretation rules
        condition_engine.py Multi-marker association logic
    templates/
        base.html           Shared navigation
        dashboard.html      Patient dashboard
        ml_studio.html      ML Studio admin (datasets, training, models, jobs)
        annotation.html     Annotation studio

app.py           Flask application factory
config.py        Configuration (paths, camera thresholds, demo mode)
requirements.txt Python dependencies
tests/
    test_analysis.py   76 unit + integration tests (Phases 0-6)
scripts/
    make_synthetic.py  Generate synthetic training images from the reference chart
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Windows, macOS, or Linux
- Raspberry Pi compatible (headless `opencv-python-headless`)

### Installation

```powershell
# 1. Create virtual environment
python -m venv venv-win
.\venv-win\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
$env:LUMINE_MODE = "development"
python app.py
```

Open http://localhost:5000

On Raspberry Pi OS with Chromium installed, run `bash run.sh` from a terminal to start LUMINE and open it in fullscreen kiosk mode. If Chromium or a desktop session is unavailable, the server starts without opening a browser; visit `http://localhost:5000` manually.

### Environment Variables

| Variable       | Default          | Purpose                                          |
|----------------|------------------|--------------------------------------------------|
| LUMINE_MODE    | production       | Set development for debug + auto-reload          |
| LUMINE_HOST    | 0.0.0.0          | Bind address                                     |
| LUMINE_PORT    | 5000             | Bind port                                        |
| SECRET_KEY     | dev-secret-key   | Flask session key — change in production         |

### Running Tests

```powershell
.\venv-win\Scripts\pytest.exe tests/test_analysis.py -v
# Expected: 76 passed
```

---

## Photo Capture Protocol

The quality of strip analysis depends almost entirely on photo consistency.
Poor photos are the primary source of misclassification.

### Camera Setup

| Parameter   | Requirement                                             |
|-------------|---------------------------------------------------------|
| Distance    | 20-30 cm from strip surface                             |
| Orientation | Strip horizontal, camera directly above (no angle)      |
| Focus       | Auto-focus allowed; confirm strip is sharp before capture |
| Resolution  | 5 MP minimum recommended; 12 MP+ preferred              |

### Lighting

> WARNING: Lighting is the single biggest variable. Use the same light source every session.

| Good                                                         | Avoid                                          |
|--------------------------------------------------------------|------------------------------------------------|
| Diffuse natural light (overcast day, north-facing window)    | Direct sunlight — causes glare and colour shift |
| Softbox or ring light at 45 degree angle                     | Phone torch pointed directly at strip           |
| Consistent indoor fluorescent / LED (same brand, colour temp)| Mixed light sources (e.g. window + lamp)        |
| White paper background under the strip                       | Colourful backgrounds that bleed into strip edges |

### Preventing Glare

- Tilt the strip up to 10 degrees relative to the camera — small angles break specular glare without distorting pad geometry.
- If glare is still detected, place a diffuse white sheet above the strip as a diffuser.
- The quality gate rejects images with glare ratio > 0.8 on the strip bounding box.

### Reading the Strip

- Read the strip at the time stated in the product insert (typically 60-120 s after dipping).
- Lay the strip flat on a clean white paper surface. Do not hold it in your hand.
- Take the photo within plus or minus 10 s of the recommended read time — colour development continues after the window.

### White Reference

LUMINE automatically samples the white control zone at the end of the strip.
If your strip does not have a white control pad:

- Place a small piece of white printer paper (at least 1 cm squared) immediately adjacent to the last pad before photographing.
- In the New Screening flow, manually indicate the white reference region using the corner markers.

---

## ML Training Workflow

The ML pipeline supplements (not replaces) the Lab baseline.
A trained model is only used when explicitly activated via the ML Studio.

### Step 1 — Collect Real Samples

1. Photograph strips with **known** standard solutions (e.g., Chek-Stix, Quanti-Chrom standards).
2. Upload images via ML Studio -> Datasets -> Upload Samples.
3. Mark each sample ground-truth level in ML Studio -> Datasets -> Annotate.

### Step 2 — Generate Synthetic Supplements (Optional)

```powershell
# Generate 50 synthetic images per level for glucose on URS-4K
python scripts/make_synthetic.py --strip-type URS-4K --marker glucose --per-level 50 --out-dir uploads/synthetic
# Then upload the folder via ML Studio and mark as synthetic
```

> CAUTION: Activating a model trained entirely on synthetic images is blocked by the
> synthetic-only guard. You must include real samples, or explicitly override the guard
> in the activation modal with full acknowledgement.

### Step 3 — Train

In ML Studio -> Training:

- Select strip type, marker, algorithm (rf = Random Forest, gb = Gradient Boosting).
- Pick one or more datasets.
- Click Start Training. A background subprocess trains the model and reports progress.
- Training artifacts are saved to models/ as .joblib files with SHA-256 hash verification.

### Step 4 — Evaluate and Activate

1. Open ML Studio -> Models.
2. Click Metrics on the new model — review validation accuracy, macro-F1, and confusion matrix.
3. If val accuracy beats the Lab baseline (target: 70% or higher on real data), click Activate.
4. Future /api/analyze calls for that (strip_type, marker) pair will use the ML model.
5. To revert: click Deactivate — the system falls back to the Lab baseline immediately.

### Minimum Data Requirements

| Classes per marker     | Minimum real photos per class |
|------------------------|-------------------------------|
| 2 (negative/positive)  | 5                             |
| 4-6 (typical)          | 5 per class (20-30 total)     |

The training pipeline enforces min_per_class=5 by default and raises a clear error if violated.

---

## API Reference

### Analysis

| Method | Endpoint              | Description                                                                  |
|--------|-----------------------|------------------------------------------------------------------------------|
| POST   | /api/analyze          | Analyse an uploaded image. Body: {image_path, patient_id, sample_type, test_profile} |
| POST   | /api/camera/capture   | Capture a frame from the attached camera                                     |
| GET    | /api/camera/preview   | Get a live JPEG preview frame                                                |

### Datasets

| Method | Endpoint                                       | Description                        |
|--------|------------------------------------------------|------------------------------------|
| GET    | /api/datasets                                  | List all datasets                  |
| POST   | /api/datasets                                  | Create dataset {name, strip_type, description} |
| GET    | /api/datasets/<id>                             | Dataset metadata                   |
| POST   | /api/datasets/<id>/samples                     | Upload a sample image (multipart)  |
| GET    | /api/datasets/<id>/samples                     | List samples                       |
| GET    | /api/datasets/<id>/export/csv                  | Export labelled pads as CSV        |
| GET    | /api/datasets/<id>/samples/<sid>/labels        | Get pad labels for a sample        |
| POST   | /api/datasets/<id>/samples/<sid>/labels        | Save pad labels                    |
| DELETE | /api/datasets/<id>                             | Delete dataset and all samples     |

### Training

| Method | Endpoint                                    | Description                               |
|--------|---------------------------------------------|-------------------------------------------|
| POST   | /api/training/jobs                          | Launch a training job                     |
| GET    | /api/training/jobs                          | List recent jobs                          |
| GET    | /api/training/jobs/<id>                     | Poll job status and log output            |
| GET    | /api/training/models                        | List models (strip_type, marker, active_only params) |
| GET    | /api/training/models/<id>                   | Model metadata + metrics                  |
| POST   | /api/training/models/<id>/activate          | Activate model for inference              |
| POST   | /api/training/models/<id>/deactivate        | Deactivate — rollback to baseline         |
| DELETE | /api/training/models/<id>                   | Delete inactive model record              |

---

## Known Limitations and Validation Status

This section is an honest disclosure of what has and has not been validated.

### What Is Verified

| Component                          | How verified                                                         |
|------------------------------------|----------------------------------------------------------------------|
| CIE Lab colour space conversion    | Unit tests against known sRGB-to-Lab reference values                |
| CIEDE2000 delta-E calculation      | Cross-checked against published worked examples                      |
| Strip/pad detection                | 76/76 unit and integration tests pass                                |
| White reference normalisation      | Shift-vector unit tests; manual visual inspection                    |
| Quality gate thresholds            | Blur / exposure / glare unit tests                                   |
| ML training pipeline (synthetic)   | End-to-end fit-save-load-predict cycle verified on synthetic data     |
| Database schema and CRUD           | 76/76 tests covering all tables and lifecycle operations             |
| REST API routing                   | Syntax checks; import checks; route registration verified            |

### What Requires Real-World Validation

| Component                              | Gap                                                                  |
|----------------------------------------|----------------------------------------------------------------------|
| Strip detector accuracy                | Not tested against a systematic real-photo dataset. MSER-based contour detection may fail with unusual lighting or strip brands. |
| Pad grid alignment                     | Uniform-grid assumption fails if pads are not evenly spaced (some multi-analyte strips vary spacing). |
| Lab baseline accuracy                  | Reference colours derived from manufacturer colour charts, not from photographed strips. Expect systematic bias without calibration. |
| ML model performance                   | No real training data collected yet. Pipeline is correct but untrained on real photos. |
| White reference auto-detection         | May fail if the control zone is discoloured, partially cropped, or absent on some strip brands. |
| Inter-phone / inter-lighting reproducibility | Not measured. Colour accuracy degrades with different cameras or lighting conditions. |
| Clinical concordance                   | Results have NOT been compared against laboratory reference values for real patient samples. |

### Recommended Validation Steps Before Use

1. **Photo capture study**: Collect 50+ strip photos of Chek-Stix or equivalent standard solutions under varying lighting. Measure strip detection success rate.
2. **Colour calibration**: Use a ColorChecker card in each session to post-correct camera colour profiles before ML training.
3. **Baseline accuracy study**: Compare LUMINE Lab baseline predictions against known-level standard strips for all markers. Record per-marker confusion matrices.
4. **ML training run**: With 5+ real photos per class per marker, train and evaluate. Accept only if val macro-F1 is 0.80 or higher.
5. **Clinical concordance study**: With appropriate institutional approval, compare 50+ real patient results (LUMINE vs. laboratory) before any clinical use.

### Security Notes

- Change SECRET_KEY in production.
- The upload endpoint accepts arbitrary JPEG/PNG. Validate MIME type and file size in production.
- DEMO_MODE=True adds a visible DEMO badge but does not restrict any functionality.
- The SQLite database is stored unencrypted. Do not store real patient data without appropriate encryption and access controls.

---

## Directory Layout

```
LUMINE/
    app.py                           Flask application factory
    config.py                        App configuration
    requirements.txt                 Python dependencies
    README.md                        This file
    buildlumineplan.md               Original build plan
    finishpartfromthebuildplan.md    Implementation tracker (Phases 0-7)
    app/
        database/        SQLite schema and DB manager
        image_processing/ Strip/pad detection, quality gates, white reference
        ml/              Feature extraction, training pipeline, baseline predictor
        health/          Marker interpretation and condition association
        static/          CSS, JS, images
        templates/       Jinja2 HTML templates
    scripts/
        make_synthetic.py  Synthetic training image generator
    tests/
        test_analysis.py   76 tests (Phases 0-6)
    models/              Trained .joblib artifacts (SHA-256 verified)
    uploads/             Captured and uploaded strip images
    calibration/         Camera calibration files
    docs/
        audit.md         Implementation audit log
    logs/                Application logs
```#   L U M I N E  
 