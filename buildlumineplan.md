# LUMINE — Strip Analysis & ML Integration Build Plan (v2, agent-ready)

**checking the done parts** check first the finishpartfrombuildplan.md to see if there is any part of the plan that is already done. if there is, please skip it. if you are done with a part, please list them in the finishpartfrombuildolan.md so that we can track what is done on the builplan 


**System:** LUMINE Intelligent Health Screening Platform (Flask)
**Goal:** Replace the placeholder strip analysis with real computer vision, add a reference-color baseline, then add dataset management and model training/activation — in that order.
**Status of this document:** Revised after review. Section 0 lists what changed and why.

---

## 0. Review Notes (what was wrong in v1)

| # | Problem in v1 | Fix in v2 |
|---|---|---|
| 1 | Audit claims cite exact line numbers that cannot be trusted to still match the code | Phase 0 makes the agent **re-verify every claim** before changing anything |
| 2 | "Random" in the title is inaccurate: the current code is deterministic (fixed center crop + fixed slicing), just wrong | Reworded to "placeholder heuristics" |
| 3 | Plan assumes **10 pads** everywhere, but the strip types are named `URS-4K` / `SALIVA-4K` (suggesting 4 parameters?) | Pad count and markers must be **read from strip definitions in the DB/config**, never hardcoded. Agent must confirm and report |
| 4 | ML was the centerpiece, but there is no data, and the models/ directory is empty. With little data, ML will be *worse* than reference-chart matching | Reordered: **calibrated Lab chart-matching is the baseline and default**. ML is activated only if it beats the baseline on held-out data |
| 5 | "In-browser training" with epochs/loss curves, but Random Forest / Gradient Boosting have no epochs or loss curves | Training UI shows what the chosen model actually produces (CV scores, confusion matrix, feature importance). Loss curves only for the optional MLP |
| 6 | Training in a "background thread" with progress in memory breaks under multi-worker servers (gunicorn) and blocks on the GIL | Training runs as a **separate subprocess**, progress stored in the DB/file, UI polls (SSE optional) |
| 7 | Pickle loading is a remote-code-execution risk if model files can ever be uploaded or tampered with | Models saved with joblib only by the server's own trainer, in a non-web-served dir, with SHA-256 recorded in the DB and verified on load. **No model file upload endpoint** |
| 8 | Train/val split by pad or sample would leak (pads from the same photo in both sets) | Split **by strip photo** (group split), stratified where possible |
| 9 | One pad = one chemistry. "Confusion matrix for each of the 10 pads" implies one model, which is unclear | **One model per (strip_type, marker)**, trained only on that marker's pads |
| 10 | Strip detection via Otsu fails on white strips over light backgrounds; no failure path | Quality gate: if detection is not confident, **return an error/retake request** instead of guessing. Plus manual corner-adjust fallback in UI |
| 11 | Color-card/white-balance mentioned in the architecture but never actually specified | Defined: use the strip's white substrate/holder as white reference + record capture metadata. Optional printed card later |
| 12 | `ultralytics` (YOLOv8) is AGPL-licensed and heavy; `torch` conflicts with the Raspberry Pi/CPU-only goal | Both **removed from scope**. OpenCV first. Optional extras documented separately |
| 13 | `colour-science` is a heavy dependency just to compute ΔE | Use `scikit-image` (`skimage.color.rgb2lab`, `deltaE_ciede2000`) |
| 14 | Model storage inconsistent (`<model_name>.pkl` vs `active_model.pkl`), no versioning or rollback | Model registry table with versions, metrics, hashes, active flag, one-click rollback |
| 15 | "Pre-packaged starter dataset with simulated strips" has no source | Replaced with a **synthetic generator** (clearly labeled synthetic, used only for tests/demo, never for activating a model) |
| 16 | Health context: nothing about safety, privacy, access control, or limits of claims | Added Section 7 (safety, privacy, security) |
| 17 | Tests were left to the final step; existing test files are empty | Tests are written **per phase** with acceptance criteria |
| 18 | `/api/analyze` response format undefined | Contract defined in Section 5 |

---

## 1. Rules for the Agent

1. **Verify before you change.** Run Phase 0 first. If anything in Section 2 is false, report the real state and adapt; do not follow the plan blindly.
2. **Work phase by phase.** Finish a phase's acceptance criteria, run its tests, commit, then continue. Do not start Phase N+1 with Phase N failing.
3. **Do not hardcode** pad count, marker names, or reference colors. Read them from the strip definition/DB.
4. **Never fabricate results.** If detection or confidence is low, return a "retake photo" outcome, not a guess.
5. **Minimal dependencies.** Only add packages listed in Section 6. Pin versions in `requirements.txt`.
6. **Keep the app running.** Each phase must leave the existing app starting and `/api/analyze` responding.
7. **Ask the user** (stop and report) if: the strip definitions disagree with the plan, real strip photos are needed, or a decision in Section 8 blocks you.
8. At the end of each phase, output a short report: files changed, tests run + results, known limitations.

---

## 2. Current-State Findings to Verify (Phase 0)

These were observed in an earlier audit. Line numbers may be stale: confirm by reading the code.

1. **`app/image_processing/strip_detection.py`**: believed to crop a fixed central rectangle (~70% of the image) with no real detection.
2. **`app/image_processing/pad_detection.py`**: believed to split that rectangle into a fixed number of equal slices (`num_pads = 10`).
3. **`app/routes.py` (urine analysis path)**: `db.get_reference_colors('URS-4K', marker)` believed to return `[]`, so the route falls back to hardcoded grayscale reference values.
4. **`app/ml/chart_colors.py`**: believed to contain real chart colors that are never imported.
5. **`models/`** is believed empty; **`tests/test_analysis.py`, `test_camera.py`, `test_health.py`** are believed to be 0 bytes.

**Also determine (not in v1):**
- How many pads/markers does each strip type (`URS-4K`, `SALIVA-4K`) actually have? Where is that defined?
- Are the colors in `chart_colors.py` sRGB values? Are they for the strips actually in use?
- How does the app run (dev server, gunicorn, number of workers, Raspberry Pi vs laptop)? This affects the training design.
- How are users/roles handled (is there login)? Needed for Section 7.
- Where are uploaded images stored now?

**Deliverable of Phase 0:** a short `docs/audit.md` with confirmed/denied findings and the answers above, plus a green `pytest` run (even if only a smoke test).

---

## 3. Target Architecture

```
Capture/Upload
   -> Quality checks (blur, exposure, size)
   -> Strip localization (OpenCV) + perspective rectification
   -> Pad segmentation (positions from strip definition, refined by image)
   -> White-reference normalization
   -> Per-pad features (Lab-centered)
   -> Predictor:  [active ML model for (strip, marker)]  OR  [Lab chart matching baseline]
   -> Result + confidence + warnings + debug overlay
```

Training side (admin only):

```
Dataset (photos + pad labels) -> group split by photo -> feature extraction
 -> train per (strip, marker) in subprocess -> metrics vs baseline
 -> register model version -> activate only if it beats baseline
```

Key principle: **the baseline must work well on its own.** ML is an upgrade, not a prerequisite.

---

## 4. Implementation Phases

### Phase 1 — Color references + Lab baseline
**Do**
- Seed reference colors from `app/ml/chart_colors.py` into the DB for each strip type/marker/level (idempotent seed, safe to re-run).
- Remove the hardcoded grayscale fallback in `routes.py`. If references are missing for a marker, return a clear error for that marker.
- Create `app/ml/color.py`: sRGB→Lab (D65) and CIEDE2000 using `scikit-image`.
- Create `app/ml/baseline.py`: nearest reference color by ΔE00; return level, ΔE to best and second-best match, and a confidence derived from the margin between them.

**Acceptance**
- Unit test: a pad whose RGB exactly equals a reference color returns that level with ΔE ≈ 0.
- Unit test: ordering/ties and out-of-gamut inputs handled.
- No code path uses the old grayscale fallback.

### Phase 2 — Strip & pad localization (computer vision)
**Do**
- `strip_detection.py`: find the strip with contour detection (try HSV/saturation and edge-based segmentation, not only Otsu, since strips are often white on light surfaces). Use `cv2.minAreaRect` + `warpPerspective` to produce an upright, axis-aligned strip image. Handle vertical/horizontal/rotated.
- Return a **detection confidence** (area ratio, aspect ratio vs expected strip aspect, edge strength). Below threshold → raise `StripNotFound`.
- `pad_detection.py`: use the strip definition (pad count, expected relative positions) as a prior, then refine along the strip axis with an intensity/color profile. Sample each pad from an **inner region** (e.g., central 50–60% of the pad) to avoid edges and bleed.
- Compute per-pad dispersion; flag pads with high variance (uneven coloring, glare).
- Output a **debug overlay image** (strip outline + pad boxes) for the UI and for tests.
- Add manual fallback: UI lets the user drag 4 corners if auto-detection fails (can ship as a later sub-task, but the API must accept optional corner points).

**Acceptance**
- Synthetic test images (generated in tests): strip at 0°, 30°, 90°, off-center, on dark and light backgrounds → pad centers found within a defined tolerance.
- Image with no strip → `StripNotFound`, not a guess.
- If real sample photos exist in the repo or the user supplies them, run them and report results with overlays.

### Phase 3 — White reference, quality gates, and analyze integration
**Do**
- White reference: sample the strip's white substrate (between/around pads) or holder; compute per-channel gain to normalize illumination before Lab conversion. Document its limits.
- Quality gates: blur (variance of Laplacian), over/under-exposure (clipped pixel fraction), glare (saturated blobs on pads), minimum strip resolution. Each produces a warning or a retake error.
- Wire into `/api/analyze` using the baseline predictor. Implement the response contract in Section 5.

**Acceptance**
- Tests for each quality gate (blurred, dark, overexposed, glare images).
- End-to-end test: synthetic strip built from reference colors → analyze → expected levels returned for every pad.
- Existing UI still works with the new response (update frontend only as required).

### Phase 4 — Dataset management & annotation
**Do**
- Tables (with migrations): `training_datasets`, `training_samples` (photo-level: image path, strip type, capture metadata, split), `training_pad_labels` (sample id, marker, pad index, true level, optional measured value, annotator, timestamp). Labels are per pad, linked to a photo.
- Routes: create dataset, upload sample, get sample, save pad labels, list/export dataset (CSV + zip). Validate uploads (Section 7).
- Annotation page: show the auto-detected overlay, allow adjusting boxes/corners, choose the true level for each pad. Records whether labels came from a known standard solution/chart or visual judgment (**label source** field).
- Synthetic generator script (`scripts/make_synthetic.py`) for tests and demos. Synthetic samples carry `is_synthetic = true` and are excluded from activation decisions.

**Acceptance**
- Round trip test: create dataset → upload image → save labels → export → counts match.
- Group split helper: no photo appears in both train and validation.

### Phase 5 — Training engine
**Do**
- `app/ml/features.py`: per-pad features: mean and median Lab, HSV, normalized chromaticity (r, g), difference from white reference, pad dispersion. Versioned (`FEATURE_VERSION`) so old models are rejected if features change.
- `app/ml/trainer.py`: per `(strip_type, marker)`: Random Forest and Gradient Boosting (scikit-learn). Use grouped, stratified cross-validation by photo; report accuracy, macro-F1, per-class recall, confusion matrix, **mean absolute level error** (levels are ordinal, so being off by one is better than off by three), and feature importance. Handle class imbalance (`class_weight="balanced"`). Refuse to train with fewer than a minimum samples per class (configurable, default e.g. 10) and say why.
- Compare against the Phase 1 baseline on the same validation photos.
- Run training in a **subprocess** (e.g., `multiprocessing`/`subprocess` calling a CLI entry point). Store job status and log lines in the DB or a job file so any worker can serve progress.
- Save models with `joblib` into a non-public directory `models/<strip>/<marker>/v<N>.joblib`. Record in table `ml_models`: strip, marker, version, algorithm, feature version, dataset ids, metrics JSON, baseline metrics JSON, SHA-256, created by, `is_active`.

**Acceptance**
- Test on synthetic data: fit → evaluate → save → load (hash verified) → predict.
- Test: training refuses on too-little data; corrupted file fails the hash check.
- Test: the same photo never lands in both train and validation.

### Phase 6 — ML Studio UI + activation
**Do**
- `/ml_studio` (admin only): dataset library, annotation entry, training form (strip type, marker, algorithm, estimators, etc.), job progress, and results (metrics, confusion matrix, feature importance, comparison to baseline).
- **Activation rules:** the "Activate" button is enabled only if the model's validation metric is at least as good as the baseline **and** the data is not purely synthetic. Admin can override with an explicit confirmation that is logged. Rollback = activate a previous version.
- Inference: `/api/analyze` uses the active model for each (strip, marker) if available; otherwise baseline. The response states which method was used.

**Acceptance**
- Manual test script in the docs: train → view metrics → activate → analyze → rollback.
- Tests: non-admin gets 403 on training/activation routes; activation guard blocks a worse model.

### Phase 7 — Hardening and documentation
- Full test run, lint, update README (setup, how to collect good photos: even lighting, no glare, strip flat, fixed distance, same phone if possible).
- Document known limits and the validation protocol (Section 7).
- Final report to the user: what works, what is unverified, what data is still needed.

---

## 5. `/api/analyze` Response Contract

```json
{
  "status": "ok | retake | error",
  "method": "ml | baseline",
  "strip_type": "URS-4K",
  "quality": { "blur": 0.0, "exposure": "ok", "glare": false, "warnings": [] },
  "detection": { "confidence": 0.0, "overlay_image": "<url or base64>" },
  "results": [
    {
      "marker": "...",
      "pad_index": 0,
      "level": "negative | trace | +1 | ...",
      "confidence": 0.0,
      "delta_e_best": 0.0,
      "delta_e_second": 0.0,
      "low_confidence": false
    }
  ],
  "model": { "id": null, "version": null },
  "disclaimer": "Screening aid only. Not a diagnosis."
}
```

`status: "retake"` includes a human-readable reason. Keep old fields if the current frontend depends on them (check first, then migrate).

---

## 6. Dependencies

| Need | Package | Notes |
|---|---|---|
| Web | Flask (existing) | |
| CV | `opencv-python-headless`, `numpy` | headless for servers/Pi |
| Color | `scikit-image` | `rgb2lab`, `deltaE_ciede2000` |
| ML | `scikit-learn`, `joblib` | no GPU needed |
| Tests | `pytest` | |

**Out of scope for this plan:** `torch`, `ultralytics`/YOLO, `colour-science`. Revisit only if classical methods demonstrably fail on real photos.

---

## 7. Safety, Privacy, Security, Validation

- **Not a diagnostic device.** Show the disclaimer on every result. Do not use wording such as "diagnosis"; use "screening result." Low-confidence results must be visibly marked.
- **Real-world validation is required before trusting results:** compare against strips read by known standards or a reference method on real photos across lighting conditions and phones. Report accuracy honestly. Synthetic accuracy proves nothing about real performance.
- **Access control:** only authorized (admin/researcher) users can open `/ml_studio`, label data, train, or activate models. Log who activated what.
- **Data privacy:** images may be linked to people. Store minimal metadata, no names in file names, document consent and retention, provide delete-sample/dataset.
- **Upload security:** allow only JPEG/PNG, enforce max size and dimensions, re-encode images with OpenCV/Pillow, use `secure_filename`/random names, store outside static web root, protect state-changing routes with CSRF tokens or equivalent.
- **Model files:** server-generated only, no upload route, hash-verified on load.

---

## 8. Open Questions for the User (agent should surface, not guess)

1. Real number of pads/markers for `URS-4K` and `SALIVA-4K`, and do the reference colors in `chart_colors.py` match those exact strips?
2. Are there real labeled strip photos available? How many, taken with which phone/lighting? (Needed for Phases 4–6 to be meaningful.)
3. Is there user login/roles? If not, how should `/ml_studio` be protected?
4. Deployment target: laptop, Raspberry Pi, or hosted server?
5. Is the intended use research/demo or something that will be used on real people? (Affects the validation and disclaimer requirements.)

---

## 10. YOLOv8 Strip Localization Integration (Raspberry Pi 5 Optimization)

### Dataset & Model Suitability
- **Is the YOLOv8 PyTorch dataset good for YOLOv8-Nano?**
  - **Yes, absolutely.** A dataset formatted for YOLOv8 PyTorch (containing images and YOLO normalized bounding box `.txt` annotations) is the exact format used to train **YOLOv8-Nano (`yolov8n`)**.
  - Since URS-10T and URS-4K strips share identical white plastic strip geometry, a detector trained on strip boxes detects URS-4K test strips seamlessly.

### Architecture & Raspberry Pi 5 Deployment Strategy
1. **Model Size:** Train or fine-tune using **YOLOv8-Nano (`yolov8n`)**, the smallest and fastest model variant (~3.2M params).
2. **ONNX Export:** Export the trained weights to ONNX format (`yolov8n.onnx`).
3. **OpenCV DNN Inference (Zero heavy dependencies on Pi):**
   - Run inference on Raspberry Pi 5 using `cv2.dnn.readNetFromONNX()`.
   - **No PyTorch or Ultralytics needed in production on the Raspberry Pi.**
   - Keeps runtime memory footprint under 70MB and inference latency between ~30–50ms on Pi 5 CPU.
4. **Classical CV Fallback:**
   - Retain current Phase 2 multi-cue contour detector as an automated fallback if YOLO detection confidence falls below the detection threshold.

### ML Studio Features to Add for YOLOv8
1. **Upload Dataset `.zip`:** Dedicated upload endpoint and UI to accept Roboflow YOLOv8 exports (zipped containing `data.yaml`, `train/`, `valid/`, `test/`).
2. **Auto-extract & Validation:** Server verifies directory structure, parses `data.yaml` classes, and counts image-label pairs.
3. **Train Strip Detector Button:**
   - Background training trigger for `yolov8n`.
   - Automated export of best weights to `yolov8n.onnx`.
   - Model registry tracking (ONNX path, mAP metrics, SHA-256 hash).
   - One-click activate/deactivate in `/ml_studio`.

---

## 11. Milestones & Definition of Done

| Milestone | Phases | Done when |
|---|---|---|
| M0 | 0 | Audit verified, test harness runs |
| M1 | 1–3 | Analyze works on rotated/off-center strips with Lab baseline, quality gates, tests green |
| M2 | 4 | Datasets can be created, annotated, exported; group split tested |
| M3 | 5 | Per-marker models train, evaluate against baseline, register with hashes |
| M4 | 6 | Admin-only ML Studio; guarded activation and rollback working |
| M5 | 7 | Docs complete, final report states validation status honestly |
| M6 | 8 | YOLOv8-nano strip detection with ONNX/cv2.dnn on RPi 5 + CV fallback |

**M1 is the most valuable deliverable.** If time is short, stop there: a correct baseline is better than an unvalidated model.

