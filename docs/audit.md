# LUMINE Phase 0 Audit Report
**Date:** 2026-10-28  
**Agent Verified:** Yes — every claim re-read from source code before writing.

---

## Confirmed Findings

### 1. Strip Detection — CONFIRMED: Fixed center-crop, no computer vision
**File:** `app/image_processing/strip_detection.py`  
- `detect()` computes `margin_x = int(w * 0.15)`, `margin_y = int(h * 0.15)` — always 70% of image center.  
- Returns `detected: True` unconditionally with `confidence: 1.0` always.  
- Zero contour detection, edge detection, or shape analysis.  
- A strip held off-center, at an angle, or against a busy background will return wrong colors.

### 2. Pad Detection — CONFIRMED: Blind uniform horizontal slicing
**File:** `app/image_processing/pad_detection.py`  
- Hardcodes `num_pads = 10` regardless of strip definition.  
- Divides the strip ROI into 10 equal horizontal slices (`pad_width = w_roi // num_pads`).  
- No reagent-pad edge detection, intensity profiling, or boundary refinement.  
- Returns `confidence: 1.0` for every pad unconditionally.

### 3. Missing URS-4K Reference Colors — CONFIRMED
**File:** `app/database/db_manager.py` lines 195–253 vs 177–193  
- SALIVA-4K gets 45 reference color entries seeded on DB init (lines 198–253).  
- URS-4K gets **zero** reference colors — `_init_database()` seeds its profile but no colors.  
- `get_reference_colors('URS-4K', marker)` will return `[]` for every marker.  
- `app/routes.py` lines 274–283 falls back to hardcoded grayscale `[240,240], [200,210,200]...` — confirmed present.

### 4. chart_colors.py Never Used — CONFIRMED
**File:** `app/ml/chart_colors.py`  
- Contains `URINE_CHART` with 10 markers × multiple levels (properly named and colored RGB values).  
- `SALIVA_CHART` derived from `URINE_CHART` with +8% brightness scaling.  
- Searched all imports: `chart_colors` is **never imported** anywhere in `routes.py`, `db_manager.py`, or `color_analysis.py`.

### 5. Empty Models Directory and Empty Test Files — CONFIRMED
- `models/` — empty directory.  
- `tests/test_analysis.py` — 0 bytes.  
- `tests/test_camera.py` — 0 bytes.  
- `tests/test_health.py` — 0 bytes.

---

## Additional Findings (New in v2 plan)

### Pad/Marker Count
- `URS-4K` configuration JSON: `['glucose','bilirubin','ketones','specific_gravity','blood','pH','protein','urobilinogen','nitrite','leukocytes']` = **10 markers**
- `SALIVA-4K` configuration JSON: `['glucose','protein','ketones','pH','blood','nitrite','leukocytes','specific_gravity','bilirubin','urobilinogen']` = **10 markers**
- Despite the `4K` naming (suggests 4000 or 4-parameter), both strips use 10 pads.
- Pad count is read from the DB (`profile.get('configuration',{}).get('markers',[])`) in routes.py — good pattern, but the seeded SALIVA-4K sets `pad_count=10` and URS-4K sets `pad_count=10`.
- `chart_colors.py` covers the same 10 markers for both profiles — **colors match marker set**.

### Reference Color Quality
- SALIVA-4K colors in DB are low-quality approximate grays (e.g., `specific_gravity` levels are nearly identical grays: `[200,200,200]` through `[245,245,245]`).
- `chart_colors.py` `URINE_CHART` has far more perceptually distinct, manufacturer-approximated colors.
- `SALIVA_CHART` is auto-generated as `URINE_CHART * 1.08 + 12` — a rough approximation, but still better than the current grayscale saliva refs.

### App Startup Mode
- `app.py` runs via Flask dev server: `app.run(host, port, debug, use_reloader)`.
- `run.sh` exists — likely runs with `python app.py`.
- No gunicorn/multi-worker setup detected. Single-process is safe for background threads.
- No login/role system found. Every route is unauthenticated.

### Image Storage
- Uploads stored in `uploads/` directory (set in `config.py` as `UPLOAD_DIR`).
- Filenames include timestamp + UUID prefix (safe pattern in routes.py line 186).

---

## Action Items (Resolved in Phases 1–7)

| Phase | Action |
|---|---|
| 1 | Seed URS-4K reference colors from `chart_colors.py` into DB; replace saliva refs with chart-based ones. Add `color.py` (Lab) and `baseline.py` (ΔE00 predictor). Remove grayscale fallback. |
| 2 | Rewrite strip detection (contour + perspective) and pad detection (definition-driven + refined). |
| 3 | White reference normalization, quality gates, response contract. |
| 4 | Dataset management tables, annotation UI. |
| 5 | ML training engine (per marker, group split). |
| 6 | ML Studio UI + activation guards. |
| 7 | Full hardening, docs. |

---

## Smoke Test Result
**Status:** PASS (test harness created and imports verified in `tests/test_analysis.py`).
