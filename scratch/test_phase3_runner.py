import sys
import os
import cv2
import numpy as np
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.ml.chart_colors import URINE_CHART
from app.image_processing import QualityGateChecker, WhiteReferenceNormalizer

# 1. Quality Gates Tests
q_checker = QualityGateChecker()

# Blurred image
sharp_img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(sharp_img, "TEST LUMINE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
blurred_img = cv2.GaussianBlur(sharp_img, (45, 45), 0)
q_blur = q_checker.evaluate_image(blurred_img)
assert q_blur['retake_required'] is True, "Blurred image must trigger retake"
print("PASS: Quality gate caught blurred image (blur score: {:.2f})".format(q_blur['blur']))

# Underexposed (dark) image
dark_img = np.full((400, 400, 3), 10, dtype=np.uint8)
q_dark = q_checker.evaluate_image(dark_img)
assert q_dark['retake_required'] is True, "Dark image must trigger retake"
assert q_dark['exposure'] == 'underexposed'
print("PASS: Quality gate caught dark image")

# Overexposed image
bright_img = np.full((400, 400, 3), 252, dtype=np.uint8)
q_bright = q_checker.evaluate_image(bright_img)
assert q_bright['retake_required'] is True, "Overexposed image must trigger retake"
assert q_bright['exposure'] == 'overexposed'
print("PASS: Quality gate caught overexposed image")

# Glare image
glare_img = np.full((400, 400, 3), 100, dtype=np.uint8)
cv2.circle(glare_img, (200, 200), 120, (255, 255, 255), -1)
q_glare = q_checker.evaluate_image(glare_img)
assert q_glare['glare'] is True, "Specular glare must be detected"
assert q_glare['retake_required'] is True
print("PASS: Quality gate caught glare image")

# 2. White Reference Normalization Test
normalizer = WhiteReferenceNormalizer(target_white=(245.0, 245.0, 245.0))
# Strip under warm yellow light: R=250, G=240, B=180
rect_strip_warm = np.full((60, 480, 3), (180, 240, 250), dtype=np.uint8) # BGR
pads_dummy = [{'color': [200, 180, 100]}]
norm_pads, meta = normalizer.normalize_strip_pads(rect_strip_warm, pads_dummy)
gains = meta['gains_rgb']
# Gain for B should be significantly higher than gain for R
assert gains[2] > gains[0], "Blue channel should receive higher gain under warm illumination"
print("PASS: White reference gains computed successfully: {}".format(gains))

# 3. End-to-end Synthetic Strip Analyze Test
# Use the exact DB marker sequence for URS-4K
spec = importlib.util.spec_from_file_location("app_main", os.path.join(ROOT, "app.py"))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)
flask_app = app_mod.create_app()
client = flask_app.test_client()

db = flask_app.db
profile = db.get_test_profile('URS-4K')
markers = profile['configuration']['markers']
expected_levels = {}
pad_colors_rgb = []
for m in markers:
    lvl, rgb = URINE_CHART[m][0]
    expected_levels[m] = lvl
    pad_colors_rgb.append(rgb)

strip_len, strip_w = 400, 40
img = np.full((600, 800, 3), 40, dtype=np.uint8)
strip_canvas = np.full((strip_w, strip_len, 3), (245, 245, 245), dtype=np.uint8)

for i, rgb in enumerate(pad_colors_rgb):
    px = int(i * 30 + 10)
    bgr = (rgb[2], rgb[1], rgb[0])
    cv2.rectangle(strip_canvas, (px, 6), (px + 22, 34), bgr, -1)

x0, y0 = 200, 280
img[y0:y0+strip_w, x0:x0+strip_len] = strip_canvas

test_img_path = os.path.join(ROOT, "scratch", "test_synth_strip.jpg")
cv2.imwrite(test_img_path, img)



resp = client.post('/api/analyze', json={
    'image_path': test_img_path,
    'patient_id': 'TEST_P001',
    'sample_type': 'urine',
    'test_profile': 'URS-4K'
})

data = resp.get_json()
assert data['status'] == 'ok', "Expected status ok, got {}".format(data.get('status'))
assert data['method'] == 'baseline'
assert 'quality' in data
assert 'blur' in data['quality']
assert 'exposure' in data['quality']
assert 'glare' in data['quality']
assert 'detection' in data
assert data['detection']['confidence'] > 0.5
assert len(data['results']) == 10

correct = 0
for res in data['results']:
    m = res['marker']
    pred = res['level']
    exp = expected_levels[m]
    if pred == exp:
        correct += 1
    print("{:16s}: Pred={:10s} Exp={:10s} (dE={:.2f}, conf={:.2f})".format(
        m, pred, exp, res.get('delta_e_best', 0.0), res.get('confidence', 0.0)
    ))

print("Accuracy: {}/{}".format(correct, len(markers)))
assert correct == 10, "All 10 pads must match expected levels"
print("ALL PHASE 3 RUNNER TESTS PASSED!")
