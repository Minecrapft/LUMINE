# LUMINE Urine Strip Selection and Capture Build Plan

## Goal

Let a user choose the exact urine strip profile before taking or uploading a photo.
Analyze the image using that profile and show only the parameters physically present
on that strip.

The requested four-parameter strip has:

- Glucose (`glucose`)
- pH (`pH`)
- Protein (`protein`)
- Ketones (`ketones`)

This is a research-use workflow. It must not present results as a clinical diagnosis.

## Implementation status

The selected URS-4K profile now represents the user's physical four-pad strip:

- `URS-4K` has four configured pads in the selected order: Glucose, pH, Protein,
  Ketones. Analysis and synthetic labels use this profile configuration.
- `URS-10T` remains a separate ten-marker profile for compatibility, but is not
  offered in New Screening.
- The four-pad reference colors are a subset of LUMINE's generic urine chart.
  They are approximate, not manufacturer-specific or clinically validated; the UI
  identifies the profile as experimental/research-use.
- The New Screening page lists profiles from the database, and the camera page shows
  the chosen profile and parameters.
- Camera capture and image upload both submit the selected profile to analysis.
- Analysis validates the profile and returns only its configured marker results.
  Patient-history trend cards now derive markers from recorded results instead of
  showing a fixed set.
- Missing profile IDs are rejected by the camera and analysis routes instead of
  silently defaulting to SALIVA-4K.
- Added focused tests for profile seeding, selection, profile binding, photo upload,
  and the four-result output shape.

The exact strip model, physical pad order, and manufacturer chart remain to be
confirmed. Until then, treat the results as experimental only.

## Additional requested work — patient management and result images

**Status: IMPLEMENTED**

### Patient deletion with screening history

- Add a clearly visible Delete action for each patient in the Patients page.
- Require a confirmation step that identifies the patient and warns that their
  screening history will also be deleted.
- Delete the patient's assessments and dependent records consistently, then remove
  the patient. Use a database transaction and explicit foreign-key/dependency
  handling; do not leave orphaned records or image files.
- If related files are removed, delete only paths proven to be inside the app's
  configured upload directory. Never delete arbitrary paths from database rows.
- Surface failures to the user and retain records if deletion does not complete.
- Test cascade/dependency handling, success, unknown patient IDs, and failure
  behavior.

### Start screening for a specific patient

- When starting a screening from a patient-specific action, pass that patient's ID
  into the New Screening workflow.
- Preselect and display that patient on step 1 so the user does not have to search
  again. Keep the ability to change the patient.
- Preserve the current general New Screening flow when no patient ID is supplied.
- Test valid, missing, and unknown patient IDs.

### Display source and analyzed images with results

- Show the original captured/uploaded image on the results page.
- Show the model/pad-detection visualization (such as the generated overlay with
  strip/pad outlines) as a separate image, clearly labeled so users can distinguish
  source photo from analysis visualization.
- Persist paths or URLs in the assessment safely, retain backward compatibility for
  older assessments that have no image/overlay, and present a useful empty state.
- Do not describe the overlay as a model prediction if it only visualizes detection;
  use the actual analysis method and available artifacts to label it accurately.
- Ensure saved image URLs cannot expose arbitrary server files.
- Test that both images are shown for new results, and that older results render
  without them.

### Show assessment images and prediction confidence in patient history

- For each screening in the patient's timeline, show the original captured/uploaded
  image and the associated annotated analysis image when safely available.
- Show confidence for each stored marker prediction, formatted as a percentage. Do
  not invent a confidence score for older assessments where it was not stored.
- Label the score as model/predictor match confidence, not a medical probability or
  diagnostic certainty.
- If an image or confidence is unavailable, state that rather than showing a broken
  image or misleading zero value.

### Implementation and verification

- Patient deletion now confirms the patient/history deletion in the UI and removes
  assessments plus their audit entries transactionally. Uploaded originals and
  generated overlays are removed only when their resolved paths are inside the
  configured upload directory. The API reports missing patients and file-cleanup
  warnings.
- Patient-specific New Screening opens with that patient already selected, while
  retaining the ability to change patients and the regular no-query workflow.
- Results show the original photo and a separate analysis image. The analysis image
  shows detected pad regions plus predicted readings associated with pad indices.
  It is accurately labeled as an analysis visualization, not a separately
  AI-generated/reconstructed image.
- Assessment image URLs are served through an assessment-scoped route that only
  returns existing files under the configured upload directory. Older assessments
  without images display an empty-state message.
- Changed files: `app/database/db_manager.py`, `app/routes.py`,
  `app/templates/patients.html`, `app/templates/new_screening.html`,
  `app/templates/results.html`, `tests/test_analysis.py`, and this plan.
- Verification: focused patient/profile workflow tests pass (3 passed); Python
  compilation passes for the changed Python files. The patient tests cover
  preselection, delete confirmation UI, history/audit deletion, safe image cleanup,
  blocking outside-upload image paths, transactional failure preserving patient,
  history, and files, source/overlay rendering, and legacy assessments with no
  image.
- Limitation: the analysis visualization overlays the app's computed readings and
  pad detections on the captured image. It is not a new image generated by a neural
  network. The urine profile remains experimental and its color references are
  approximate pending the actual product chart and pad-order confirmation.
- Patient history now shows each assessment's original image, annotated analysis
  image, and stored per-marker prediction confidence where available. Confidence is
  identified as predictor match confidence, not diagnostic certainty; legacy
  assessments without a stored score say it is unavailable.
- Additional history implementation files: `app/templates/history.html`,
  `app/routes.py`, and `tests/test_analysis.py`.
- Additional verification: focused profile/history tests pass (3 passed), including
  real confidence persistence in the response and unavailable-confidence handling
  for legacy records; Python compilation passes.

## Current application behavior

- The New Screening wizard already has a test-profile selection step and navigates to
  the camera with a `test_profile` query parameter (`app/templates/new_screening.html`).
- The camera sends that profile to `/api/analyze`
  (`app/templates/camera.html`).
- `/api/analyze` reads the profile's `configuration.markers` and uses the number of
  markers to detect pads (`app/routes.py`).
- The original URS-4K profile has ten markers; the new four-marker profile is a
  separate database profile.
- Analysis results are stored on the assessment and displayed from the stored
  `marker_results` in `app/templates/results.html`.

Therefore, the main issue is not adding profile selection from scratch. It is
registering the requested four-parameter strip as a distinct, correctly configured
profile and ensuring the selected profile stays authoritative across capture,
analysis, storage, and display.

## Scope and requirements

1. **Profile selection**
   - Present the user with available test profiles and their sample type and marker
     list.
   - Make the requested four-parameter urine strip selectable before capture.
   - Do not silently use a default profile when the requested profile is missing or
     invalid; show a useful error and prevent analysis.
   - Keep existing saliva and urine profiles available unless the user explicitly
     chooses to retire one.

2. **Profile definition**
   - Define the requested profile with exactly `glucose`, `pH`, `protein`, and
     `ketones`, in the physical pad order for the actual strip.
   - Derive pad count from the configured marker list rather than hardcoding four in
     the capture or analysis code.
   - Use a distinct profile ID (working name: `URS-4K-4P`) until the exact product
     identity is confirmed. Do not overwrite the existing `URS-4K` ten-marker
     definition or its reference data as part of this feature.
   - Keep labels, casing, and marker IDs consistent with existing chart colors,
     training records, marker interpretation, and stored assessments.

3. **Capture and analysis**
   - Carry the selected profile ID from the wizard to the camera and `/api/analyze`.
   - Validate the selected profile and its marker configuration on the server.
   - Use that profile's ordered marker list for pad detection and prediction.
   - Return exactly one result per configured marker, in pad order. Do not append
     generic markers that are not on the selected strip.
   - Continue to reject/retake unclear or poor-quality images rather than fabricating
     missing readings.

4. **Results and persistence**
   - Save the selected profile ID with the assessment.
   - Show the selected strip/profile and only its configured marker results in the
     results page, history, and any relevant API output.
   - Ensure summaries, health-domain calculations, recommendations, and exports do
     not add unsupported markers or treat absent markers as negative.
   - Preserve old assessments as recorded; changing a profile must not rewrite
     historical results.

5. **Photo upload**
   - Keep camera capture and image upload on the same selected-profile workflow.
   - Include the selected profile with uploaded images and use it for analysis.
   - Show the profile name/parameter list before the user submits a photo so they 
     catch a wrong strip selection.

## Implementation plan

### Phase 1 — Verify the real strip profile

- Confirm the exact strip packaging/model name and the physical order of the four
  pads.
- Check the existing database profile, chart references, and any model/training data
  for collisions with the proposed profile ID.
- Confirm each marker's supported result levels and whether valid reference colors
  exist for that strip.
- Do not claim clinical validity based only on the current profile's `validated`
  label; retain the research-use disclaimer.

**Gate:** the experimental profile is available for software testing, but do not
rely on its readings or represent it as manufacturer-accurate until its physical pad
order, product identity, and reference chart are confirmed. If those differ from
the current provisional configuration, update the profile and tests.

### Phase 2 — Register and expose profiles

- Add the four-marker profile using the project's existing test-profile storage and
  initialization patterns.
- Ensure initialization safely handles both fresh databases and databases that
  already contain the old ten-marker URS-4K profile.
- Provide profile data to the New Screening UI from the server/database instead of
  maintaining a separate hardcoded list in the HTML.
- Display the chosen profile's name and parameters before capture.

### Phase 3 — Enforce the selected profile end to end

- Validate the profile ID and marker list on the server.
- Keep camera capture, upload, and analysis bound to the same selected profile.
- Confirm pad detection uses the configured marker count and pad order.
- Ensure analysis, interpretation, saved assessments, result display, history, and
  exports only include the selected profile's markers.
- Handle missing references or unsupported markers as explicit errors; do not make
  up values or substitute another profile's reference chart.

### Phase 4 — Tests and verification

Add or update tests to verify:

- The profile list includes the four-marker urine strip and reports the right
  parameters.
- Selecting that profile sends its profile ID into capture and analysis.
- Analysis requests exactly four pads and returns exactly four results in configured
  order.
- Results do not contain bilirubin, blood, specific gravity, urobilinogen, nitrite,
  or leukocytes for this four-marker profile.
- Unknown profile IDs and profiles with invalid/empty marker definitions fail
  explicitly.
- The existing ten-marker URS-4K and saliva profile behavior is unchanged.
- Existing assessments remain readable and keep their original profile/results.
- Both captured and uploaded photos use the selected profile.

Run the narrow relevant test suite first, then the project's full test suite.
Manually smoke-test the New Screening → select urine strip → capture/upload → results
flow.

## Expected outcome

A user chooses the four-parameter urine strip before taking or uploading a photo.
LUMINE processes only its four configured pads and presents only Glucose, pH,
Protein, and Ketones. Other strips can be selected using their own profiles, without
changing the meaning of the four-parameter strip or existing records.

## Confirmed profile decisions and remaining limitation

- The user's `URS-4K` strip has four pads: Glucose, pH, Protein, and Ketones.
- The configured pad order is Glucose → pH → Protein → Ketones.
- Manufacturer-specific reference colors/result levels have not been supplied.
  Existing generic references must not be presented as manufacturer-validated.

## Agent instructions

- Read the current routes, templates, database schema/initialization, marker
  interpretation, chart references, and tests before editing; code may have changed
  since this plan was written.
- Preserve existing profile behavior and avoid unrelated refactors.
- Implement in phases, run tests, and report files changed, test results, and any
  unresolved profile/reference assumptions.
- Do not create or imply clinical validation. Keep the research-use limitations
  visible.

- add what you finished in the website in the finishpartfromthebuildplan.md so we can track what we finished