"""
LUMINE Test Suite — Phase 0 smoke tests + Phase 1 acceptance tests.
Run with: pytest tests/test_analysis.py -v
"""
import sys
import os
import json
import uuid
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import app modules without installing the package
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ===========================================================================
# Phase 0 — Smoke tests: key modules importable, DB usable
# ===========================================================================

class TestPhase0Smoke:
    """Phase 0: every critical module can be imported; DB initialises cleanly."""

    def test_flask_app_module_importable(self):
        """The app module must be parseable without opening the camera."""
        import importlib.util, sys
        # Just check the file parses without error by loading its spec
        spec = importlib.util.spec_from_file_location(
            "app_main",
            os.path.join(ROOT, "app.py")
        )
        assert spec is not None, "app.py not found"
        # We intentionally do not exec_module to avoid application initialization side-effects

    def test_create_app_does_not_open_camera(self, tmp_path, monkeypatch):
        import importlib.util

        class FakeCamera:
            def __init__(self):
                self.initialize_calls = 0

            def initialize(self):
                self.initialize_calls += 1
                return True

        spec = importlib.util.spec_from_file_location(
            "lumine_app_factory_test",
            os.path.join(ROOT, "app.py")
        )
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        monkeypatch.setattr(app_module, "CAMERA_AVAILABLE", True)
        monkeypatch.setattr(app_module, "CameraManager", FakeCamera)

        app = app_module.create_app({
            "TESTING": True,
            "DATABASE_PATH": str(tmp_path / "factory.db")
        })

        assert isinstance(app.camera, FakeCamera)
        assert app.camera.initialize_calls == 0

    def test_chart_colors_importable(self):
        from app.ml.chart_colors import URINE_CHART, SALIVA_CHART, PROFILE_CHARTS
        assert "glucose" in URINE_CHART
        assert "glucose" in SALIVA_CHART
        assert "pH" in PROFILE_CHARTS["SALIVA-PH4"]
        # Every marker has at least 2 levels
        for marker, levels in URINE_CHART.items():
            assert len(levels) >= 2, f"{marker} has fewer than 2 levels"

    def test_chart_colors_ten_markers(self):
        from app.ml.chart_colors import URINE_CHART
        assert len(URINE_CHART) == 10, f"Expected 10 markers, got {len(URINE_CHART)}"

    def test_db_init_and_reference_query(self, tmp_path):
        """URS-4K has four pads while URS-10T retains its ten-marker profile."""
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        # Both profiles should exist
        profile_urine = db.get_test_profile("URS-4K")
        profile_urine_10t = db.get_test_profile("URS-10T")
        profile_urine_4p = db.get_test_profile("URS-4K-4P")
        profile_saliva = db.get_test_profile("SALIVA-4K")
        profile_saliva_ph = db.get_test_profile("SALIVA-PH4")
        assert profile_urine is not None, "URS-4K profile missing"
        assert profile_urine_10t is not None, "URS-10T profile missing"
        assert profile_urine_4p is not None, "four-marker urine profile missing"
        assert profile_saliva is not None, "SALIVA-4K profile missing"
        assert profile_saliva_ph is not None, "SALIVA-PH4 profile missing"
        # Each profile should have markers configured
        urine_markers = profile_urine.get("configuration", {}).get("markers", [])
        urine_10t_markers = profile_urine_10t.get("configuration", {}).get("markers", [])
        saliva_markers = profile_saliva.get("configuration", {}).get("markers", [])
        assert urine_markers == ["glucose", "pH", "protein", "ketones"]
        assert profile_urine.get("pad_count") == 4
        assert profile_urine.get("validation_status") == "experimental"
        assert len(urine_10t_markers) == 10
        assert len(saliva_markers) == 10, f"SALIVA-4K: expected 10 markers, got {len(saliva_markers)}"
        assert profile_saliva_ph.get("sample_type") == "saliva"
        assert profile_saliva_ph.get("pad_count") == 4
        assert profile_saliva_ph.get("reaction_time") == 0
        assert profile_saliva_ph.get("validation_status") == "experimental"
        assert profile_saliva_ph.get("configuration", {}).get("markers") == ["pH"]
        assert len(db.get_reference_colors("SALIVA-PH4", "pH")) == 15
        for marker in urine_10t_markers:
            assert db.get_reference_colors("URS-10T", marker)
        for marker in urine_markers:
            assert db.get_reference_colors("URS-4K", marker)
        assert profile_urine_4p.get("sample_type") == "urine"
        assert profile_urine_4p.get("configuration", {}).get("markers") == [
            "glucose", "pH", "protein", "ketones"
        ]
        assert profile_urine_4p.get("pad_count") == 4
        for marker in profile_urine_4p["configuration"]["markers"]:
            assert db.get_reference_colors("URS-4K-4P", marker)

    def test_existing_urs4k_profile_is_migrated_to_four_pads(self, tmp_path):
        import json as _json
        from app.database.db_manager import DatabaseManager

        db_path = str(tmp_path / "profile_migration.db")
        db = DatabaseManager(db_path)
        with db.get_connection() as conn:
            conn.execute(
                """
                UPDATE test_profiles SET pad_count = 10, configuration = ?
                WHERE profile_id = 'URS-4K'
                """,
                (_json.dumps({
                    "markers": [
                        "glucose", "bilirubin", "ketones", "specific_gravity",
                        "blood", "pH", "protein", "urobilinogen", "nitrite", "leukocytes"
                    ]
                }),)
            )
            conn.commit()

        migrated_db = DatabaseManager(db_path)
        profile = migrated_db.get_test_profile("URS-4K")
        assert profile["pad_count"] == 4
        assert profile["configuration"]["markers"] == [
            "glucose", "pH", "protein", "ketones"
        ]

    def test_dashboard_stats_ignore_orphaned_assessments(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "stats.db"))
        patient_id = db.create_patient({"first_name": "Current", "last_name": "Patient"})
        db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K",
            "overall_status": "followup_recommended"
        })
        db.create_assessment({
            "patient_id": "P_SYNTH_001",
            "sample_type": "urine",
            "test_profile": "URS-4K",
            "overall_status": "followup_recommended"
        })

        stats = db.get_stats()
        assert stats["total_patients"] == 1
        assert stats["total_assessments"] == 1
        assert stats["followup_needed"] == 1
        assert stats["recent_screening_count"] == 1
        assert stats["status_counts"] == {"followup_recommended": 1}

    def test_strip_detection_initialization(self):
        """Confirm StripDetector can be imported and initialized."""
        from app.image_processing.strip_detection import StripDetector, StripNotFound
        det = StripDetector()
        assert det is not None
        dummy = np.zeros((100, 200, 3), dtype=np.uint8)
        with pytest.raises(StripNotFound):
            det.detect(dummy)

    def test_pad_detection_initialization(self):
        """Confirm PadDetector can be imported and initialized."""
        from app.image_processing.pad_detection import PadDetector
        det = PadDetector()
        assert det is not None
        dummy = np.zeros((100, 200, 3), dtype=np.uint8)
        pads = det.detect_pads(dummy, (0, 0, 200, 100), num_pads=10)
        assert len(pads) == 10, f"Expected 10 pads, got {len(pads)}"

    def test_screening_selector_offers_supported_sample_profiles(self, tmp_path):
        from io import BytesIO
        from flask import Flask
        import cv2
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp
        import app.routes as routes

        app = Flask(
            "lumine_test",
            template_folder=os.path.join(ROOT, "app", "templates")
        )
        app.db = DatabaseManager(str(tmp_path / "routes.db"))
        app.camera = None
        app.register_blueprint(main_bp)
        client = app.test_client()
        patient_id = app.db.create_patient({"first_name": "Test", "last_name": "User"})

        dashboard_patient_id = app.db.create_patient({
            "first_name": "Dashboard",
            "last_name": "Patient"
        })
        app.db.create_assessment({
            "patient_id": dashboard_patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K",
            "overall_status": "followup_recommended"
        })
        app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K",
            "overall_status": "normal"
        })

        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert b"Health Screening Dashboard" in dashboard.data
        assert b"Good ," not in dashboard.data
        assert b'href="/health_insights" class="view-all">View insights' in dashboard.data
        assert b"Screenings in the last 7 days" in dashboard.data
        assert b"Follow-up recommended" in dashboard.data
        assert b"Health Domains" not in dashboard.data

        insights = client.get("/health_insights")
        assert insights.status_code == 200
        assert b'class="assessment-item insight-assessment status-followup_recommended needs-attention"' in insights.data
        assert b'class="assessment-item insight-assessment status-normal"' in insights.data
        assert b"Within range" in insights.data

        screening = client.get("/new_screening")
        assert screening.status_code == 200
        assert b"Camera offline" not in screening.data
        assert b'id="nav-ml-studio"' not in screening.data
        assert b'data-test="URS-4K"' in screening.data
        assert b"URS-4K Urine Strip (4 parameters)" in screening.data
        assert b'data-test="URS-10T"' not in screening.data
        assert b"URS-4K-4P" not in screening.data
        assert b"SALIVA-4K" not in screening.data
        assert b'data-test="SALIVA-PH4"' in screening.data
        assert b'<span class="sample-name">Saliva</span>' in screening.data
        assert b'data-sample="saliva" aria-pressed="false"' in screening.data

        system = client.get("/system")
        assert system.status_code == 200
        assert b"Open ML Studio" in system.data
        assert b'href="/ml_studio"' in system.data
        ml_studio = client.get("/ml_studio")
        assert ml_studio.status_code == 200
        assert b'href="/system" class="nav-link active"' in ml_studio.data

        camera_10t = client.get("/camera?test_profile=URS-10T&sample_type=urine")
        assert camera_10t.status_code == 200
        assert b"URS-10T Urine Strip" in camera_10t.data
        assert b"Glucose, Bilirubin, Ketones" in b" ".join(camera_10t.data.split())

        camera_4k = client.get("/camera?test_profile=URS-4K&sample_type=urine")
        assert camera_4k.status_code == 200
        assert b"Glucose, pH, Protein, Ketones" in b" ".join(camera_4k.data.split())
        assert b'id="lightBrightness"' in camera_4k.data
        assert b"Strip light" in camera_4k.data
        assert b'id="focusSlider"' in camera_4k.data
        assert b"Manual focus" in camera_4k.data
        assert b"Image Capture" not in camera_4k.data
        assert b"quality-checks" not in camera_4k.data
        assert b"Lighting" not in camera_4k.data
        assert b"Focus" not in camera_4k.data
        assert b"Position" not in camera_4k.data
        assert b'id="cameraPreview"' in camera_4k.data

        camera_saliva_ph = client.get("/camera?test_profile=SALIVA-PH4")
        assert camera_saliva_ph.status_code == 200
        assert b"all four pads visible" in b" ".join(camera_saliva_ph.data.split())
        assert b"one pH value" in b" ".join(camera_saliva_ph.data.split())

        camera = client.get("/camera?test_profile=URS-4K-4P&sample_type=saliva")
        assert camera.status_code == 200
        assert b"4-Parameter Urine Strip" in camera.data
        assert b"Glucose, pH, Protein, Ketones" in b" ".join(camera.data.split())

        unknown = client.get("/camera?test_profile=UNKNOWN")
        assert unknown.status_code == 404
        missing_profile = client.get("/camera")
        assert missing_profile.status_code == 400

        invalid_analysis = client.post(
            "/api/analyze",
            json={
                "patient_id": patient_id,
                "test_profile": "UNKNOWN",
                "image_path": "unused.jpg"
            }
        )
        assert invalid_analysis.status_code == 404
        missing_profile_analysis = client.post(
            "/api/analyze", json={"patient_id": patient_id, "image_path": "unused.jpg"}
        )
        assert missing_profile_analysis.status_code == 400

        class FakeQualityChecker:
            def evaluate_image(self, image):
                return {
                    "retake_required": False, "reason": "", "blur": 100,
                    "exposure": "good", "glare": False, "warnings": []
                }

            def evaluate_strip_resolution(self, strip):
                return True, ""

        class FakeStripDetector:
            def __init__(self, **kwargs):
                pass

            def detect(self, image, manual_corners=None):
                return {"confidence": 1.0, "rectified_strip": None, "method": "test"}

        class FakePadDetector:
            def detect_pads(self, image, strip, num_pads, **kwargs):
                assert num_pads == 4
                from app.ml.chart_colors import URINE_CHART
                markers = ["glucose", "pH", "protein", "ketones"]
                return [{"color": URINE_CHART[marker][0][1]} for marker in markers]

            def generate_debug_overlay(self, image, strip, pads):
                return image

        class FakeWhiteNormalizer:
            def normalize_strip_pads(self, strip, pads):
                return pads, {}

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(routes, "QualityGateChecker", FakeQualityChecker)
        monkeypatch.setattr(routes, "StripDetector", FakeStripDetector)
        monkeypatch.setattr(routes, "PadDetector", FakePadDetector)
        monkeypatch.setattr(routes, "WhiteReferenceNormalizer", FakeWhiteNormalizer)
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        app.config["UPLOAD_DIR"] = str(upload_dir)

        success, encoded = cv2.imencode(".png", np.full((100, 100, 3), 128, dtype=np.uint8))
        assert success
        analyzed = client.post(
            "/api/analyze",
            data={
                "patient_id": patient_id,
                "test_profile": "URS-4K-4P",
                "image": (BytesIO(encoded.tobytes()), "strip.png")
            },
            content_type="multipart/form-data"
        )
        monkeypatch.undo()
        assert analyzed.status_code == 200
        result = analyzed.get_json()
        assert result["status"] == "ok"
        assert [marker["marker"] for marker in result["results"]] == [
            "glucose", "pH", "protein", "ketones"
        ]
        assert all(
            isinstance(marker.get("confidence"), (int, float))
            and 0 <= marker["confidence"] <= 1
            for marker in result["results"]
        )
        assessment = app.db.get_assessment(result["assessment_id"])
        overlay_path = upload_dir / ("overlay_" + os.path.basename(assessment["image_path"]))
        annotated = cv2.imread(str(overlay_path))
        assert annotated is not None
        assert annotated.shape[0] > 100
        assert client.get(
            f"/assessments/{result['assessment_id']}/image/overlay"
        ).status_code == 200

    def test_camera_is_started_on_demand_and_released_when_stopped(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp

        class FakeCamera:
            def __init__(self):
                self.is_initialized = False
                self.initialize_calls = 0
                self.release_calls = 0

            def initialize(self):
                self.initialize_calls += 1
                self.is_initialized = True
                return True

            def get_live_preview(self):
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                self.release_calls += 1
                self.is_initialized = False

        app = Flask(
            "camera_lifecycle_test",
            template_folder=os.path.join(ROOT, "app", "templates")
        )
        app.db = DatabaseManager(str(tmp_path / "camera.db"))
        app.camera = FakeCamera()
        app.register_blueprint(main_bp)
        client = app.test_client()

        assert app.camera.initialize_calls == 0
        preview = client.get("/api/camera/preview")
        assert preview.status_code == 503
        assert app.camera.initialize_calls == 0

        camera_page = client.get("/camera?test_profile=URS-4K")
        assert camera_page.status_code == 200
        assert app.camera.initialize_calls == 1

        preview = client.get("/api/camera/preview")
        assert preview.status_code == 200
        assert app.camera.initialize_calls == 1

        stream = client.get("/api/camera/stream", buffered=False)
        assert stream.status_code == 200
        assert stream.mimetype == "multipart/x-mixed-replace"
        assert stream.headers["Cache-Control"].startswith("no-store")
        first_frame = next(stream.response)
        assert first_frame.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        assert b"\xff\xd8" in first_frame
        stream.close()

        stopped = client.post("/api/camera/stop")
        assert stopped.status_code == 200
        assert app.camera.release_calls == 1
        assert not app.camera.is_initialized

    def test_camera_requests_720p_at_30_fps(self, monkeypatch):
        import app.camera.camera_manager as camera_module

        class FakeCapture:
            def __init__(self):
                self.settings = {}
                self.released = False

            def isOpened(self):
                return True

            def set(self, property_id, value):
                self.settings[property_id] = value
                return True

            def get(self, property_id):
                return self.settings.get(property_id, 0)

            def read(self):
                return True, np.zeros((720, 1280, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        capture = FakeCapture()
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *_args: capture)
        monkeypatch.setattr(camera_module.time, "sleep", lambda _seconds: None)

        camera = camera_module.CameraManager()
        assert camera.initialize() is True
        assert capture.settings[camera_module.cv2.CAP_PROP_FRAME_WIDTH] == 1280
        assert capture.settings[camera_module.cv2.CAP_PROP_FRAME_HEIGHT] == 720
        assert capture.settings[camera_module.cv2.CAP_PROP_FPS] == 30
        assert capture.settings[camera_module.cv2.CAP_PROP_AUTOFOCUS] == 1
        camera.release()

    def test_manual_focus_disables_autofocus_and_sets_focus_value(self, monkeypatch):
        import app.camera.camera_manager as camera_module

        class FakeCapture:
            def __init__(self):
                self.settings = {}

            def set(self, property_id, value):
                self.settings[property_id] = value
                return True

            def get(self, property_id):
                return self.settings.get(property_id, 0)

            def release(self):
                pass

        camera = camera_module.CameraManager()
        camera.camera = FakeCapture()
        camera.is_initialized = True

        result = camera.set_manual_focus(176)

        assert result["supported"] is True
        assert result["autofocus_disabled"] is True
        assert result["focus_value"] == 176
        assert camera.camera.settings[camera_module.cv2.CAP_PROP_AUTOFOCUS] == 0
        assert camera.camera.settings[camera_module.cv2.CAP_PROP_FOCUS] == 176.0
        camera.release()

    def test_manual_focus_endpoint_validates_and_reports_unsupported_camera(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp

        class FakeCamera:
            is_initialized = True

            def set_manual_focus(self, value):
                assert value == 150
                return {"supported": False, "requested_focus": value}

        app = Flask("camera_focus_api_test")
        app.db = DatabaseManager(str(tmp_path / "camera_focus.db"))
        app.camera = FakeCamera()
        app.register_blueprint(main_bp)
        client = app.test_client()

        invalid = client.post("/api/camera/focus", json={"value": 300})
        assert invalid.status_code == 400

        unsupported = client.post("/api/camera/focus", json={"value": 150})
        assert unsupported.status_code == 422
        assert "does not support manual focus" in unsupported.get_json()["error"]

    def test_illumination_controller_sets_pwm_and_turns_off_on_close(self):
        from app.hardware.illumination import IlluminationController

        class FakeOutput:
            def __init__(self):
                self.value = 0
                self.closed = False

            def close(self):
                self.value = 0
                self.closed = True

        output = FakeOutput()
        controller = IlluminationController(device_factory=lambda: output)

        assert controller.status() == {"available": True, "brightness": 0}
        assert controller.set_brightness(65) == {
            "available": True,
            "brightness": 65,
        }
        assert output.value == 0.65
        controller.close()
        assert output.value == 0
        assert output.closed is True

    def test_illumination_endpoint_validates_and_sets_brightness(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.hardware.illumination import IlluminationController
        from app.routes import main_bp

        class FakeOutput:
            value = 0

            def close(self):
                self.value = 0

        output = FakeOutput()
        app = Flask("illumination_api_test")
        app.db = DatabaseManager(str(tmp_path / "illumination.db"))
        app.camera = None
        app.illumination = IlluminationController(device_factory=lambda: output)
        app.register_blueprint(main_bp)
        client = app.test_client()

        assert client.get("/api/illumination/brightness").get_json()["brightness"] == 0
        invalid = client.post("/api/illumination/brightness", json={"value": 101})
        assert invalid.status_code == 400

        response = client.post("/api/illumination/brightness", json={"value": 40})
        assert response.status_code == 200
        assert response.get_json()["brightness"] == 40
        assert output.value == 0.4
        app.illumination.close()

    def test_camera_capture_saves_image_with_preview_orientation(self, tmp_path, monkeypatch):
        from flask import Flask
        import app.routes as routes

        class FakeCamera:
            is_initialized = True

            def capture_frame(self, timeout=5.0):
                image = np.zeros((3, 4, 3), dtype=np.uint8)
                image[0, 0] = (0, 0, 255)
                image[-1, -1] = (255, 0, 0)
                return image

        app = Flask("camera_capture_orientation_test")
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        app.config["UPLOAD_DIR"] = str(upload_dir)
        app.camera = FakeCamera()
        app.register_blueprint(routes.main_bp)
        written_frames = []

        def capture_write(_path, image):
            written_frames.append(image.copy())
            return True

        monkeypatch.setattr(routes.cv2, "imwrite", capture_write)

        response = app.test_client().post("/api/camera/capture")
        assert response.status_code == 200
        assert os.path.dirname(response.get_json()["image_path"]) == str(upload_dir)
        assert len(written_frames) == 1
        saved = written_frames[0]
        assert tuple(saved[0, 0]) == (255, 0, 0)
        assert tuple(saved[-1, -1]) == (0, 0, 255)

    def test_patient_workflow_deletion_and_assessment_images(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        source_image = upload_dir / "capture_test.jpg"
        overlay_image = upload_dir / "overlay_capture_test.jpg"
        source_image.write_bytes(b"source image")
        overlay_image.write_bytes(b"overlay image")
        outside_image = tmp_path / "outside.jpg"
        outside_image.write_bytes(b"do not delete outside upload directory")

        app = Flask(
            "lumine_patient_test",
            template_folder=os.path.join(ROOT, "app", "templates")
        )
        app.config["UPLOAD_DIR"] = str(upload_dir)
        app.db = DatabaseManager(str(tmp_path / "patients.db"))
        app.camera = None
        app.register_blueprint(main_bp)
        patient_id = app.db.create_patient({"first_name": "Ari", "last_name": "Patient"})
        first_assessment_id = app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "image_path": str(source_image),
            "marker_results": [
                {
                    "marker": "glucose", "result": "negative",
                    "status": "normal", "confidence": 0.83,
                    "color": [236, 255, 255],
                    "color_raw": [130, 180, 190],
                    "chart_match_level": "negative",
                    "chart_match_rgb": [142, 197, 199],
                    "color_similarity_percent": 72
                }
            ]
        })
        outside_assessment_id = app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "image_path": str(outside_image),
            "marker_results": []
        })
        legacy_assessment_id = app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "marker_results": [
                {"marker": "pH", "result": "7.0", "status": "normal"}
            ]
        })
        app.db.log_action("test", "analysis_completed", "test assessment", first_assessment_id)
        client = app.test_client()

        new_screening = client.get("/new_screening?patient_id=" + patient_id)
        assert new_screening.status_code == 200
        assert b"id=\"selectedName\">Ari Patient" in new_screening.data
        assert b"id=\"step1Next\" disabled" not in new_screening.data
        assert b'class="patient-search" style="display:none;"' in new_screening.data
        assert client.get("/new_screening?patient_id=UNKNOWN").status_code == 404

        patients_page = client.get("/patients")
        assert patients_page.status_code == 200
        assert b"btn-delete-patient" in patients_page.data

        results_page = client.get("/results/" + first_assessment_id)
        assert results_page.status_code == 200
        assert b"Original captured/uploaded photo" in results_page.data
        assert b"detected pad regions and predicted readings" in results_page.data
        assert b"72%" in results_page.data
        assert b">Image<" in results_page.data
        assert b">Chart<" in results_page.data
        assert b"Color extracted from pad image" in results_page.data
        assert b"Closest reference chart color: negative" in results_page.data
        assert b"rgb(130, 180, 190)" in results_page.data
        assert b"rgb(236, 255, 255)" not in results_page.data
        assert b"rgb(142, 197, 199)" in results_page.data
        assert results_page.data.index(b"RESEARCH USE ONLY") > results_page.data.index(b"Limitations")
        assert results_page.data.index(b"RESEARCH USE ONLY") < results_page.data.index(b"View History")
        assert client.get(
            f"/assessments/{first_assessment_id}/image/source"
        ).data == b"source image"
        assert client.get(
            f"/assessments/{first_assessment_id}/image/overlay"
        ).data == b"overlay image"
        history_page = client.get("/history/" + patient_id)
        assert history_page.status_code == 200
        assert b"Delete Screening" in history_page.data
        assert b'class="history-details"' in history_page.data
        assert b"Readings" in history_page.data
        assert b"Photos" in history_page.data
        assert b"Chart negative" in history_page.data
        assert b'class="btn-primary history-view-results"' in history_page.data
        assert b"View results" in history_page.data
        assert b"Delete Patient &amp; History" not in history_page.data
        assert f'data-assessment-id="{first_assessment_id}"'.encode() in history_page.data
        assert b"Confidence 83%" in history_page.data
        assert b"Chart negative" in history_page.data
        assert b"72%" in history_page.data
        assert b"not a medical probability" in history_page.data
        assert b"Original photo" in history_page.data
        assert b"predicted readings" in history_page.data
        assert history_page.data.count(
            f"/assessments/{first_assessment_id}/image/".encode()
        ) == 2
        assert b"Confidence unavailable" in history_page.data
        assert client.get("/results/" + legacy_assessment_id).status_code == 200
        legacy_page = client.get("/results/" + legacy_assessment_id)
        assert b"No stored image is available" in legacy_page.data
        assert client.get(
            f"/assessments/{outside_assessment_id}/image/source"
        ).status_code == 404

        deleted = client.delete("/api/patients/" + patient_id)
        assert deleted.status_code == 200
        assert deleted.get_json()["deleted_assessments"] == 3
        assert app.db.get_patient(patient_id) is None
        assert app.db.get_patient_assessments(patient_id) == []
        assert not source_image.exists()
        assert not overlay_image.exists()
        assert outside_image.exists()
        with app.db.get_connection() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE assessment_id = ?",
                (first_assessment_id,)
            ).fetchone()[0] == 0
        assert client.delete("/api/patients/" + patient_id).status_code == 404

    def test_patient_delete_database_failure_retains_records_and_images(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        image_path = upload_dir / "preserve.jpg"
        image_path.write_bytes(b"keep image if database delete fails")

        app = Flask("lumine_delete_failure_test")
        app.config["UPLOAD_DIR"] = str(upload_dir)
        app.db = DatabaseManager(str(tmp_path / "delete_failure.db"))
        app.camera = None
        app.register_blueprint(main_bp)
        patient_id = app.db.create_patient({"first_name": "Retain", "last_name": "Me"})
        app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "image_path": str(image_path),
            "marker_results": []
        })
        with app.db.get_connection() as conn:
            conn.execute('''
                CREATE TRIGGER reject_assessment_delete
                BEFORE DELETE ON assessments
                BEGIN
                    SELECT RAISE(ABORT, 'forced delete failure');
                END
            ''')
            conn.commit()

        response = app.test_client().delete("/api/patients/" + patient_id)
        assert response.status_code == 500
        assert "records were retained" in response.get_json()["error"]
        assert app.db.get_patient(patient_id) is not None
        assert len(app.db.get_patient_assessments(patient_id)) == 1
        assert image_path.exists()

    def test_delete_single_screening_preserves_patient_and_other_results(self, tmp_path):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        from app.routes import main_bp

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        remove_source = upload_dir / "remove.jpg"
        remove_overlay = upload_dir / "overlay_remove.jpg"
        keep_source = upload_dir / "keep.jpg"
        remove_source.write_bytes(b"selected source")
        remove_overlay.write_bytes(b"selected overlay")
        keep_source.write_bytes(b"other screening source")

        app = Flask("lumine_single_assessment_delete_test")
        app.config["UPLOAD_DIR"] = str(upload_dir)
        app.db = DatabaseManager(str(tmp_path / "single_delete.db"))
        app.camera = None
        app.register_blueprint(main_bp)
        patient_id = app.db.create_patient({"first_name": "Keep", "last_name": "Patient"})
        other_patient_id = app.db.create_patient({"first_name": "Other", "last_name": "Patient"})
        assessment_id = app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "image_path": str(remove_source),
            "marker_results": []
        })
        retained_id = app.db.create_assessment({
            "patient_id": patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "image_path": str(keep_source),
            "marker_results": []
        })
        other_assessment_id = app.db.create_assessment({
            "patient_id": other_patient_id,
            "sample_type": "urine",
            "test_profile": "URS-4K-4P",
            "overall_status": "normal",
            "marker_results": []
        })
        app.db.log_action("test", "analysis_completed", "remove", assessment_id)
        app.db.log_action("test", "analysis_completed", "retain", retained_id)
        client = app.test_client()

        wrong_owner = client.delete(
            f"/api/patients/{other_patient_id}/assessments/{assessment_id}"
        )
        assert wrong_owner.status_code == 404
        assert app.db.get_assessment(assessment_id) is not None

        deleted = client.delete(
            f"/api/patients/{patient_id}/assessments/{assessment_id}"
        )
        assert deleted.status_code == 200
        assert deleted.get_json()["success"] is True
        assert app.db.get_patient(patient_id) is not None
        assert app.db.get_assessment(assessment_id) is None
        assert app.db.get_assessment(retained_id) is not None
        assert app.db.get_assessment(other_assessment_id) is not None
        assert not remove_source.exists()
        assert not remove_overlay.exists()
        assert keep_source.exists()
        with app.db.get_connection() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE assessment_id = ?",
                (assessment_id,)
            ).fetchone()[0] == 0


# ===========================================================================
# Phase 1 — Lab colour conversion + CIEDE2000 baseline predictor
# ===========================================================================

class TestPhase1ColorConversion:
    """Phase 1: sRGB -> Lab (D65) conversion via app/ml/color.py."""

    def test_white_converts_to_lab(self):
        from app.ml.color import rgb_to_lab
        lab = rgb_to_lab([255, 255, 255])
        assert abs(lab[0] - 100.0) < 1.0, "White L* should be ~100"
        assert abs(lab[1]) < 2.0, "White a* should be ~0"
        assert abs(lab[2]) < 2.0, "White b* should be ~0"

    def test_black_converts_to_lab(self):
        from app.ml.color import rgb_to_lab
        lab = rgb_to_lab([0, 0, 0])
        assert abs(lab[0]) < 1.0, "Black L* should be ~0"

    def test_delta_e_zero_for_identical(self):
        from app.ml.color import rgb_to_lab, delta_e
        lab = rgb_to_lab([120, 60, 200])
        assert delta_e(lab, lab) < 1e-6

    def test_delta_e_positive(self):
        from app.ml.color import rgb_to_lab, delta_e
        lab1 = rgb_to_lab([255, 0, 0])
        lab2 = rgb_to_lab([0, 0, 255])
        assert delta_e(lab1, lab2) > 10, "Red vs blue should have large ΔE"

    def test_rgb_to_lab_accepts_ndarray(self):
        from app.ml.color import rgb_to_lab
        arr = np.array([200, 150, 100])
        lab = rgb_to_lab(arr)
        assert len(lab) == 3

    def test_rgb_clipped_before_conversion(self):
        """Values outside 0–255 should be clipped without raising."""
        from app.ml.color import rgb_to_lab
        lab = rgb_to_lab([-10, 300, 128])
        assert all(np.isfinite(v) for v in lab)


class TestPhase1Baseline:
    """Phase 1: nearest-ΔE00 baseline predictor (app/ml/baseline.py)."""

    @pytest.fixture
    def simple_refs(self):
        """Two reference entries: negative (light) and +1 (dark)."""
        return [
            {"level": "negative", "rgb": [240, 240, 240]},
            {"level": "+1",       "rgb": [100, 150, 100]},
            {"level": "+2",       "rgb": [60,  100, 60]},
        ]

    def test_exact_match_returns_zero_delta_e(self, simple_refs):
        from app.ml.baseline import predict_level
        result = predict_level([240, 240, 240], simple_refs)
        assert result["level"] == "negative"
        assert result["delta_e_best"] < 1.0

    def test_closest_color_chosen(self, simple_refs):
        from app.ml.baseline import predict_level
        # [80, 120, 80] is closer to +2 than +1
        result = predict_level([65, 105, 65], simple_refs)
        assert result["level"] == "+2"

    def test_result_has_required_keys(self, simple_refs):
        from app.ml.baseline import predict_level
        result = predict_level([200, 200, 200], simple_refs)
        for key in (
            "level", "delta_e_best", "delta_e_second",
            "color_similarity_percent", "confidence", "low_confidence"
        ):
            assert key in result, f"Missing key: {key}"

    def test_color_similarity_is_separate_from_confidence(self, simple_refs):
        from app.ml.baseline import predict_level
        exact = predict_level([240, 240, 240], simple_refs)
        distant = predict_level([0, 0, 0], simple_refs)
        assert exact["color_similarity_percent"] == 100
        assert 0 <= distant["color_similarity_percent"] < exact["color_similarity_percent"]

    def test_confidence_high_when_margin_large(self, simple_refs):
        """Exact match on one ref far from others → high confidence."""
        from app.ml.baseline import predict_level
        result = predict_level([240, 240, 240], simple_refs)
        assert result["confidence"] > 0.7

    def test_confidence_low_when_ambiguous(self):
        """Two nearly identical refs → low confidence."""
        from app.ml.baseline import predict_level
        refs = [
            {"level": "A", "rgb": [150, 150, 150]},
            {"level": "B", "rgb": [152, 152, 152]},
        ]
        result = predict_level([151, 151, 151], refs)
        assert result["low_confidence"] is True

    def test_empty_refs_raises(self):
        from app.ml.baseline import predict_level
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            predict_level([100, 100, 100], [])

    def test_single_ref_returns_it_as_best(self):
        from app.ml.baseline import predict_level
        result = predict_level([120, 120, 120], [{"level": "only", "rgb": [120, 120, 120]}])
        assert result["level"] == "only"
        assert result["delta_e_second"] is None or result["delta_e_second"] == float("inf")


# ===========================================================================
# Phase 1 — DB seeding: URS-4K reference colors now populated
# ===========================================================================

class TestPhase1DBSeeding:
    """Phase 1: both strip types must have reference colors after DB init."""

    MARKERS = [
        "glucose", "bilirubin", "ketones", "specific_gravity",
        "blood", "pH", "protein", "urobilinogen", "nitrite", "leukocytes"
    ]
    URS_4K_MARKERS = ["glucose", "pH", "protein", "ketones"]

    def test_urine_strip_has_reference_colors(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        for marker in self.URS_4K_MARKERS:
            refs = db.get_reference_colors("URS-4K", marker)
            assert len(refs) >= 2, \
                f"URS-4K marker '{marker}' has {len(refs)} refs — expected ≥ 2"

    def test_saliva_strip_has_reference_colors(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        for marker in self.MARKERS:
            refs = db.get_reference_colors("SALIVA-4K", marker)
            assert len(refs) >= 2, \
                f"SALIVA-4K marker '{marker}' has {len(refs)} refs — expected ≥ 2"

    def test_reference_colors_have_valid_rgb(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        refs = db.get_reference_colors("URS-4K", "glucose")
        for ref in refs:
            rgb = ref.get("rgb")
            assert isinstance(rgb, list) and len(rgb) == 3, \
                f"Bad RGB for {ref.get('level')}: {rgb}"
            assert all(0 <= v <= 255 for v in rgb), f"RGB out of range: {rgb}"

    def test_urs4k_chart_uses_supplied_swatch_colors(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        from app.ml.chart_colors import URS_4K_CHART, URINE_CHART

        assert URS_4K_CHART["glucose"][0][1] == [142, 197, 199]
        assert URS_4K_CHART["pH"][0][1] == [250, 166, 53]
        assert URS_4K_CHART["protein"][0][1] == [231, 237, 153]
        assert URS_4K_CHART["ketones"][0][1] == [252, 189, 153]
        assert URS_4K_CHART["glucose"] != URINE_CHART["glucose"]

        db = DatabaseManager(str(tmp_path / "urs4k_chart.db"))
        for marker, chart_levels in URS_4K_CHART.items():
            saved = {
                row["level"]: row["rgb"]
                for row in db.get_reference_colors("URS-4K", marker)
            }
            for level, rgb in chart_levels:
                assert saved[level] == rgb

    def test_reseed_updates_urs4k_but_keeps_generic_chart_unchanged(self, tmp_path):
        import json
        from app.database.db_manager import DatabaseManager
        from app.ml.chart_colors import URS_4K_CHART, URINE_CHART

        db = DatabaseManager(str(tmp_path / "chart_reseed.db"))
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE reference_colors SET rgb = ? "
                "WHERE profile_id = 'URS-4K' AND marker = 'glucose' AND level = 'negative'",
                (json.dumps([1, 2, 3]),)
            )
            conn.commit()
        db.seed_reference_colors_idempotent()

        urs4k_negative = next(
            row for row in db.get_reference_colors("URS-4K", "glucose")
            if row["level"] == "negative"
        )
        generic_negative = next(
            row for row in db.get_reference_colors("URS-10T", "glucose")
            if row["level"] == "negative"
        )
        assert urs4k_negative["rgb"] == URS_4K_CHART["glucose"][0][1]
        assert generic_negative["rgb"] == URINE_CHART["glucose"][0][1]


# ===========================================================================
# Phase 1 — End-to-end: baseline via routes (no camera, mocked image)
# ===========================================================================

class TestPhase1NoGrayscaleFallback:
    """Phase 1: grayscale fallback must not be in routes.py any more."""

    def test_grayscale_fallback_removed(self):
        """The old fallback list must not appear in routes.py."""
        routes_path = os.path.join(ROOT, "app", "routes.py")
        with open(routes_path, "r", encoding="utf-8") as f:
            content = f.read()
        # The old fallback had this exact structure
        FORBIDDEN = "{'level': 'negative', 'rgb': [240, 240, 240]}"
        assert FORBIDDEN not in content, \
            "Grayscale fallback still present in routes.py — must be removed in Phase 1"


# ===========================================================================
# Phase 2 — Strip & Pad Localization (Computer Vision)
# ===========================================================================

class TestPhase2StripAndPadDetection:
    """Phase 2: OpenCV contour detection, perspective rectification, pad localization, and overlays."""

    @staticmethod
    def _create_synthetic_strip(angle_deg=0, bg_val=40, center=(400, 300), img_size=(600, 800)):
        import math, cv2
        h_img, w_img = img_size
        img = np.full((h_img, w_img, 3), bg_val, dtype=np.uint8)
        strip_len, strip_w = 320, 36
        strip_canvas = np.full((strip_w, strip_len, 3), (245, 248, 250), dtype=np.uint8)

        pad_colors = [
            [40, 180, 80],   # green
            [220, 190, 40],  # yellow
            [200, 120, 40],  # orange
            [180, 50, 60],   # red
            [60, 120, 210],  # blue
            [160, 70, 180],  # purple
            [100, 180, 170], # teal
            [190, 170, 120], # olive
            [210, 100, 150], # pink
            [130, 90, 60]    # brown
        ]
        pad_centers_local = []
        for i in range(10):
            px = int(i * 24 + 10)
            color = pad_colors[i]
            strip_canvas[6:30, px:px+18] = color
            pad_centers_local.append((px + 9.0, 18.0))

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx_s, cy_s = strip_len / 2.0, strip_w / 2.0
        local_corners = [(-cx_s, -cy_s), (cx_s, -cy_s), (cx_s, cy_s), (-cx_s, cy_s)]
        world_corners = [[center[0] + lx*cos_a - ly*sin_a, center[1] + lx*sin_a + ly*cos_a] for lx, ly in local_corners]

        src = np.array([[0,0], [strip_len, 0], [strip_len, strip_w], [0, strip_w]], dtype=np.float32)
        dst = np.array(world_corners, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(strip_canvas, M, (w_img, h_img))
        mask = cv2.warpPerspective(np.ones((strip_w, strip_len), dtype=np.uint8)*255, M, (w_img, h_img))
        mask_3d = cv2.merge([mask, mask, mask]) > 0
        img[mask_3d] = warped[mask_3d]

        expected_centers = []
        for px, py in pad_centers_local:
            lx = px - cx_s
            ly = py - cy_s
            wx = center[0] + lx*cos_a - ly*sin_a
            wy = center[1] + lx*sin_a + ly*cos_a
            expected_centers.append((wx, wy))

        return img, world_corners, expected_centers

    def test_strip_detection_0_deg_dark_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=0, bg_val=40)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50
        assert "rectified_strip" in res
        assert res["rectified_strip"].shape[0] > 0

    def test_strip_detection_30_deg_dark_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=30, bg_val=40)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50

    def test_strip_detection_90_deg_dark_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=90, bg_val=40)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50

    def test_strip_detection_0_deg_light_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=0, bg_val=220)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50

    def test_strip_detection_30_deg_light_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=30, bg_val=220)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50

    def test_strip_detection_90_deg_light_bg(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=90, bg_val=220)
        res = det.detect(img)
        assert res["detected"] is True
        assert res["confidence"] >= 0.50

    def test_strip_not_found_on_blank_image(self):
        from app.image_processing import StripDetector, StripNotFound
        det = StripDetector()
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        with pytest.raises(StripNotFound):
            det.detect(blank)

    def test_strip_detection_manual_corners(self):
        from app.image_processing import StripDetector
        det = StripDetector()
        img = np.full((500, 500, 3), 100, dtype=np.uint8)
        corners = [[50, 100], [350, 100], [350, 140], [50, 140]]
        res = det.detect(img, manual_corners=corners)
        assert res["detected"] is True
        assert res["method"] == "manual"
        assert res["confidence"] == 1.0

    def test_pad_detection_count_and_accuracy(self):
        from app.image_processing import StripDetector, PadDetector
        strip_det = StripDetector()
        pad_det = PadDetector()
        img, _, expected = self._create_synthetic_strip(angle_deg=0, bg_val=40)
        res = strip_det.detect(img)
        pads = pad_det.detect_pads(img, res, num_pads=10)
        assert len(pads) == 10

        # Each pad center should be within 15px of an expected center
        for p in pads:
            dcx, dcy = p["center"]
            min_d = min(np.hypot(dcx - ecx, dcy - ecy) for ecx, ecy in expected)
            assert min_d < 15.0, f"Pad {p['index']} center ({dcx}, {dcy}) too far from expected: {min_d:.1f}px"

    def test_saliva_pad_boxes_cover_compact_four_pad_section(self):
        from app.image_processing import PadDetector

        strip = np.full((60, 480, 3), 245, dtype=np.uint8)
        expected_colors = [
            (60, 170, 125),
            (55, 175, 230),
            (70, 195, 245),
            (230, 190, 170),
        ]
        expected_centers = [40, 100, 160, 220]
        for center, color in zip(expected_centers, expected_colors):
            strip[18:42, center - 16:center + 16] = color

        pads = PadDetector().detect_pads(
            strip,
            {"rectified_strip": strip},
            num_pads=4,
            detect_color_regions=True,
        )

        assert len(pads) == 4
        assert all(
            abs(pad["center"][0] - center) <= 2
            for pad, center in zip(pads, expected_centers)
        )
        assert [pad["color"] for pad in pads] == [
            [color[2], color[1], color[0]] for color in expected_colors
        ]

    def test_saliva_pad_detection_does_not_guess_when_color_regions_are_missing(self):
        from app.image_processing import PadDetector

        strip = np.full((60, 480, 3), 245, dtype=np.uint8)
        pads = PadDetector().detect_pads(
            strip,
            {"rectified_strip": strip},
            num_pads=4,
            detect_color_regions=True,
        )

        assert pads == []

    def test_pad_detection_uses_trained_class_boxes(self):
        from app.image_processing import PadDetector, StripDetector

        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        image[30:70, 25:55] = (20, 80, 220)
        image[30:70, 85:115] = (40, 180, 60)

        class FakeYOLODetector:
            def detect(self, _image):
                return [
                    {"bbox": (10, 10, 140, 80), "class_id": 10, "class_name": "strip", "confidence": 0.95},
                    {"bbox": (25, 30, 30, 40), "class_id": 2, "class_name": "Glucose", "confidence": 0.92},
                    {"bbox": (85, 30, 30, 40), "class_id": 9, "class_name": "pH", "confidence": 0.88},
                    {"bbox": (0, 5, 6, 10), "class_id": 2, "class_name": "Glucose", "confidence": 0.99},
                ]

        strip_result = StripDetector(yolo_detector=FakeYOLODetector()).detect(image)
        assert strip_result["method"] == "yolov8_nano"

        pads = PadDetector().detect_pads(
            image,
            strip_result,
            class_names=[
                "Bilirubin", "Blood", "Glucose", "Ketone", "Leukocytes",
                "Nitrite", "Protein", "SpGravity", "Urobilinogen", "pH", "strip",
            ],
            marker_names=["glucose", "pH", "protein", "ketones"],
        )

        assert [pad["marker"] for pad in pads] == ["glucose", "pH", "protein", "ketones"]
        assert [pad["bbox"] for pad in pads[:2]] == [(25, 30, 30, 40), (85, 30, 30, 40)]
        assert pads[0]["color"] == [220, 80, 20]
        assert pads[1]["color"] == [60, 180, 40]
        assert [pad["method"] for pad in pads] == [
            "yolov8_nano", "yolov8_nano", "cv_fallback", "cv_fallback",
        ]

    def test_saliva_uses_active_yolo_pad_boxes_without_urine_marker_mapping(self):
        from app.image_processing import PadDetector, StripDetector

        image = np.full((120, 200, 3), 245, dtype=np.uint8)
        expected_boxes = [
            (25, 30, 18, 24),
            (55, 30, 18, 24),
            (85, 30, 18, 24),
            (115, 30, 18, 24),
        ]

        class FakeYOLODetector:
            class_names = ["Bilirubin", "Blood", "Glucose", "Ketone", "Leukocytes",
                           "Nitrite", "Protein", "SpGravity", "Urobilinogen", "pH", "strip"]

            def detect(self, _image):
                return [
                    {"bbox": (15, 15, 160, 90), "class_id": 10,
                     "class_name": "strip", "confidence": 0.95},
                    *[
                        {"bbox": box, "class_id": index, "class_name": self.class_names[index],
                         "confidence": 0.9}
                        for index, box in enumerate(expected_boxes)
                    ],
                ]

        strip = StripDetector(yolo_detector=FakeYOLODetector()).detect(image)
        pads = PadDetector().detect_pads(
            image,
            strip,
            num_pads=4,
            class_names=FakeYOLODetector.class_names,
            marker_names=None,
            detect_color_regions=True,
        )

        assert [pad["bbox"] for pad in pads] == expected_boxes
        assert [pad["method"] for pad in pads] == ["yolov8_nano"] * 4
        assert all("marker" not in pad for pad in pads)

    def test_strip_orientation_uses_pad_detections_over_background_variance(self):
        from app.image_processing import StripDetector

        detector = StripDetector()
        rectified = np.full((60, 480, 3), 230, dtype=np.uint8)
        # A noisy handle/background region can have more variance than the pads.
        checker = (np.indices((60, 240)).sum(axis=0) % 2 * 120 + 100).astype(np.uint8)
        rectified[:, 240:] = checker[:, :, None]
        for index, color in enumerate(((20, 180, 220), (40, 190, 60), (30, 200, 220), (40, 40, 180))):
            x = 25 + index * 55
            rectified[15:45, x:x + 35] = color

        transform = np.eye(3, dtype=np.float32)
        inverse = np.eye(3, dtype=np.float32)
        detections = [
            {"class_name": "glucose", "bbox": (30, 20, 25, 20)},
            {"class_name": "pH", "bbox": (85, 20, 25, 20)},
            {"class_name": "protein", "bbox": (140, 20, 25, 20)},
            {"class_name": "ketones", "bbox": (195, 20, 25, 20)},
            {"class_name": "strip", "bbox": (0, 0, 480, 60)},
        ]

        oriented, updated_transform, updated_inverse = detector._orient_strip_canonically(
            rectified,
            yolo_detections=detections,
            transform_matrix=transform,
            inverse_transform_matrix=inverse,
        )

        assert np.array_equal(oriented, rectified)
        assert np.array_equal(updated_transform, transform)
        assert np.array_equal(updated_inverse, inverse)

    def test_reversed_strip_orientation_updates_coordinate_transforms(self):
        import cv2
        from app.image_processing import StripDetector

        detector = StripDetector()
        rectified = np.zeros((60, 480, 3), dtype=np.uint8)
        transform = np.eye(3, dtype=np.float32)
        inverse = np.eye(3, dtype=np.float32)
        detections = [
            {"class_name": "glucose", "bbox": (300, 20, 25, 20)},
            {"class_name": "pH", "bbox": (350, 20, 25, 20)},
            {"class_name": "protein", "bbox": (400, 20, 25, 20)},
        ]

        oriented, updated_transform, updated_inverse = detector._orient_strip_canonically(
            rectified,
            yolo_detections=detections,
            transform_matrix=transform,
            inverse_transform_matrix=inverse,
        )
        point = np.array([[[312.5, 30.0]]], dtype=np.float32)
        canonical = cv2.perspectiveTransform(point, updated_transform)
        original = cv2.perspectiveTransform(canonical, updated_inverse)

        assert np.array_equal(oriented, cv2.rotate(rectified, cv2.ROTATE_180))
        assert canonical[0, 0, 0] < 200
        assert np.allclose(original, point)

    def test_onnx_detector_letterbox_coordinates(self, tmp_path, monkeypatch):
        import cv2
        from app.image_processing.yolo_detector import ONNXYOLODetector

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model")
        output = np.zeros((1, 5, 8400), dtype=np.float32)
        output[0, :4, 0] = [320, 260, 100, 100]
        output[0, 4, 0] = 0.9

        class FakeNet:
            def setPreferableBackend(self, _backend):
                pass

            def setPreferableTarget(self, _target):
                pass

            def setInput(self, _blob):
                pass

            def forward(self):
                return output

        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _path: FakeNet())
        detector = ONNXYOLODetector(str(model_path), class_names=["glucose"])

        detections = detector.detect(np.zeros((320, 640, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0]["bbox"] == [270, 50, 100, 100]
        assert detections[0]["class_name"] == "glucose"

    def test_pad_dispersion_and_variance_flag(self):
        from app.image_processing import StripDetector, PadDetector
        strip_det = StripDetector()
        pad_det = PadDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=0, bg_val=40)
        res = strip_det.detect(img)
        pads = pad_det.detect_pads(img, res, num_pads=10)
        for p in pads:
            assert "dispersion" in p
            assert "high_variance" in p
            assert isinstance(p["high_variance"], bool)

    def test_debug_overlay_generation(self):
        from app.image_processing import StripDetector, PadDetector
        strip_det = StripDetector()
        pad_det = PadDetector()
        img, _, _ = self._create_synthetic_strip(angle_deg=30, bg_val=40)
        res = strip_det.detect(img)
        pads = pad_det.detect_pads(img, res, num_pads=10)
        overlay = pad_det.generate_debug_overlay(img, res, pads)
        assert overlay.shape == img.shape
        assert not np.array_equal(overlay, img), "Debug overlay must contain rendered visual annotations"

    def test_analyze_route_retake_on_no_strip(self, tmp_path):
        """When an image with no strip is submitted to /api/analyze, it returns status='retake'."""
        import importlib.util, cv2
        spec = importlib.util.spec_from_file_location("app_main", os.path.join(ROOT, "app.py"))
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        flask_app = app_mod.create_app({
            "DATABASE_PATH": str(tmp_path / "no_strip_app.db"),
            "DEMO_MODE": False
        })

        # Save a sharp, well-lit background image with a circular object (no strip)
        no_strip_img = np.full((400, 400, 3), 128, dtype=np.uint8)
        cv2.circle(no_strip_img, (200, 200), 80, (50, 50, 50), -1)
        img_path = str(tmp_path / "no_strip.jpg")
        cv2.imwrite(img_path, no_strip_img)

        client = flask_app.test_client()
        resp = client.post("/api/analyze", json={
            "image_path": img_path,
            "patient_id": "P001",
            "sample_type": "urine",
            "test_profile": "URS-4K"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "retake"
        assert "Test strip could not be detected" in data["reason"]


# ===========================================================================
# Phase 3 — White Reference, Quality Gates & Analyze Integration
# ===========================================================================

class TestPhase3QualityGatesAndIntegration:
    """Phase 3: Image quality gates, white reference normalization, and end-to-end analyze integration."""

    def test_quality_gate_blurred_image(self):
        from app.image_processing import QualityGateChecker
        import cv2
        checker = QualityGateChecker()
        sharp = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(sharp, "LUMINE MEDICAL", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        blurred = cv2.GaussianBlur(sharp, (45, 45), 0)
        report = checker.evaluate_image(blurred)
        assert report["retake_required"] is True, "Blurred image must require retake"
        assert report["blur"] < 35.0, f"Blur score too high: {report['blur']}"

    def test_quality_gate_underexposed_image(self):
        from app.image_processing import QualityGateChecker
        checker = QualityGateChecker()
        dark = np.full((400, 400, 3), 12, dtype=np.uint8)
        report = checker.evaluate_image(dark)
        assert report["retake_required"] is True
        assert report["exposure"] == "underexposed"

    def test_quality_gate_overexposed_image(self):
        from app.image_processing import QualityGateChecker
        checker = QualityGateChecker()
        bright = np.full((400, 400, 3), 252, dtype=np.uint8)
        report = checker.evaluate_image(bright)
        assert report["retake_required"] is True
        assert report["exposure"] == "overexposed"

    def test_quality_gate_glare_image(self):
        from app.image_processing import QualityGateChecker
        import cv2
        checker = QualityGateChecker()
        img = np.full((400, 400, 3), 90, dtype=np.uint8)
        cv2.circle(img, (200, 200), 120, (255, 255, 255), -1)
        report = checker.evaluate_image(img)
        assert report["glare"] is True
        assert report["retake_required"] is True

    def test_quality_gate_strip_resolution(self):
        from app.image_processing import QualityGateChecker
        checker = QualityGateChecker()
        # Strip too small (under 100px)
        tiny_res = {"detected": True, "bbox": (10, 10, 40, 6)}
        ok, reason = checker.evaluate_strip_resolution(tiny_res)
        assert ok is False
        assert "Strip resolution is too low" in reason

        # Valid strip size
        valid_res = {"detected": True, "bbox": (10, 10, 350, 30)}
        ok, reason = checker.evaluate_strip_resolution(valid_res)
        assert ok is True
        assert reason is None

    def test_white_reference_normalization(self):
        from app.image_processing import WhiteReferenceNormalizer
        normalizer = WhiteReferenceNormalizer(target_white=(245.0, 245.0, 245.0))
        # Warm incandescent lighting: high R, low B (BGR: [180, 240, 250])
        rect_strip = np.full((60, 480, 3), (180, 240, 250), dtype=np.uint8)
        pads = [{"color": [200, 180, 100]}]
        norm_pads, meta = normalizer.normalize_strip_pads(rect_strip, pads)
        gains = meta["gains_rgb"]
        # Blue channel gain must be higher than Red channel gain to balance warm cast
        assert gains[2] > gains[0]
        assert norm_pads[0]["color_raw"] == [200, 180, 100]
        assert norm_pads[0]["color"] != norm_pads[0]["color_raw"]
        assert "measured_white_rgb" in meta
        assert "target_white_rgb" in meta

    def test_end_to_end_synthetic_strip_analyze(self, tmp_path):
        """End-to-end test: synthetic strip with reference colors -> analyze -> expected levels."""
        import importlib.util, cv2
        from app.ml.chart_colors import get_chart

        spec = importlib.util.spec_from_file_location("app_main", os.path.join(ROOT, "app.py"))
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        flask_app = app_mod.create_app({
            "DATABASE_PATH": str(tmp_path / "synthetic_analysis_app.db"),
            "DEMO_MODE": False
        })
        client = flask_app.test_client()

        db = flask_app.db
        profile = db.get_test_profile("URS-4K")
        markers = profile["configuration"]["markers"]

        expected_levels = {}
        pad_colors_rgb = []
        for m in markers:
            lvl, rgb = get_chart("URS-4K")[m][0]
            expected_levels[m] = lvl
            pad_colors_rgb.append(rgb)

        # Draw a synthetic dipstick image
        strip_len, strip_w = 400, 40
        img = np.full((600, 800, 3), 40, dtype=np.uint8)
        strip_canvas = np.full((strip_w, strip_len, 3), (245, 245, 245), dtype=np.uint8)

        pad_pitch = (strip_len * 0.75 - strip_len * 0.02) / len(markers)
        pad_w = int(pad_pitch * 0.68)
        pad_h = int(strip_w * 0.62)
        for i, rgb in enumerate(pad_colors_rgb):
            center_x = strip_len * 0.02 + (i + 0.5) * pad_pitch
            px = int(center_x - pad_w / 2)
            py = int((strip_w - pad_h) / 2)
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(strip_canvas, (px, py), (px + pad_w, py + pad_h), bgr, -1)

        x0, y0 = 200, 280
        img[y0:y0+strip_w, x0:x0+strip_len] = strip_canvas

        synth_path = str(tmp_path / "synth_strip_test.jpg")
        cv2.imwrite(synth_path, img)

        resp = client.post("/api/analyze", json={
            "image_path": synth_path,
            "patient_id": "P_SYNTH_001",
            "sample_type": "urine",
            "test_profile": "URS-4K"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["method"] in {"cv_contour", "yolov8_nano"}
        assert "quality" in data
        assert "blur" in data["quality"]
        assert "exposure" in data["quality"]
        assert "glare" in data["quality"]
        assert "white_reference" in data["quality"]
        assert "detection" in data
        assert data["detection"]["confidence"] >= 0.5
        assert len(data["results"]) == len(markers) == 4

        # Verify all four configured markers predict the exact expected level.
        for res in data["results"]:
            m = res["marker"]
            assert res["level"] == expected_levels[m], (
                f"Marker {m}: expected {expected_levels[m]}, got {res['level']} (delta_e={res.get('delta_e_best')})"
            )
            assert res["delta_e_best"] < 3.0, f"Delta E too large: {res.get('delta_e_best')}"
            assert res["low_confidence"] is False
            assert res["chart_match_level"] == expected_levels[m]
            assert res["chart_match_rgb"] == get_chart("URS-4K")[m][0][1]
            assert isinstance(res["color"], list) and len(res["color"]) == 3
            assert 97 <= res["color_similarity_percent"] <= 100


# ===========================================================================
# Phase 4 — Dataset management, annotation, synthetic generator
# ===========================================================================

class TestPhase4DatasetDB:
    """Phase 4: DB CRUD for training_datasets / training_samples / training_pad_labels."""

    def _db(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        return DatabaseManager(str(tmp_path / "test.db"))

    def test_create_and_get_dataset(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("Batch A", "URS-4K", description="Test run")
        assert dsid.startswith("DS_")
        ds = db.get_dataset(dsid)
        assert ds["name"] == "Batch A"
        assert ds["strip_type"] == "URS-4K"

    def test_list_datasets_counts(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D1", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img.jpg", "URS-4K", is_synthetic=True)
        db.save_pad_labels(sid, [
            {"marker": "glucose", "pad_index": 0, "true_level": "negative", "label_source": "chart"}
        ])
        datasets = db.list_datasets()
        assert len(datasets) >= 1
        row = next(d for d in datasets if d["dataset_id"] == dsid)
        assert int(row["sample_count"]) == 1
        assert int(row["label_count"])  == 1

    def test_add_and_get_sample(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D2", "SALIVA-4K")
        meta = {"phone": "Pixel 7", "lighting": "indoor"}
        sid  = db.add_sample(dsid, "/tmp/strip.jpg", "SALIVA-4K",
                              capture_meta=meta, is_synthetic=False)
        assert sid.startswith("SMP_")
        s = db.get_sample(sid)
        assert s["strip_type"] == "SALIVA-4K"
        assert s["capture_meta"]["phone"] == "Pixel 7"
        assert s["is_synthetic"] == 0

    def test_save_and_get_pad_labels(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D3", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img2.jpg", "URS-4K", is_synthetic=True)
        labels = [
            {"marker": "glucose",  "pad_index": 0, "true_level": "negative",  "label_source": "chart"},
            {"marker": "protein",  "pad_index": 6, "true_level": "+1",        "label_source": "visual"},
            {"marker": "pH",       "pad_index": 5, "true_level": "7.0",
             "measured_value": "7.0", "label_source": "standard_solution"},
        ]
        saved = db.save_pad_labels(sid, labels, annotator="tester")
        assert saved == 3
        got = db.get_pad_labels(sid)
        assert len(got) == 3
        pad_markers = {lbl["marker"] for lbl in got}
        assert pad_markers == {"glucose", "protein", "pH"}

    def test_upsert_labels(self, tmp_path):
        """Saving labels for the same (sample, marker, pad_index) replaces old value."""
        db = self._db(tmp_path)
        dsid = db.create_dataset("D4", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img3.jpg", "URS-4K", is_synthetic=True)
        db.save_pad_labels(sid, [{"marker": "glucose", "pad_index": 0, "true_level": "negative"}])
        db.save_pad_labels(sid, [{"marker": "glucose", "pad_index": 0, "true_level": "+1"}])
        got = db.get_pad_labels(sid)
        glucose_labels = [l for l in got if l["marker"] == "glucose"]
        assert len(glucose_labels) == 1
        assert glucose_labels[0]["true_level"] == "+1"

    def test_delete_sample_removes_labels(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D5", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img4.jpg", "URS-4K")
        db.save_pad_labels(sid, [{"marker": "glucose", "pad_index": 0, "true_level": "negative"}])
        db.delete_sample(sid)
        assert db.get_sample(sid) is None
        assert db.get_pad_labels(sid) == []

    def test_delete_dataset_cascade(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D6", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img5.jpg", "URS-4K")
        db.save_pad_labels(sid, [{"marker": "glucose", "pad_index": 0, "true_level": "+2"}])
        db.delete_dataset(dsid)
        assert db.get_dataset(dsid) is None
        assert db.get_sample(sid) is None
        assert db.get_pad_labels(sid) == []

    def test_export_csv_rows(self, tmp_path):
        db = self._db(tmp_path)
        dsid = db.create_dataset("D7", "URS-4K")
        sid  = db.add_sample(dsid, "/tmp/img6.jpg", "URS-4K", is_synthetic=True)
        db.save_pad_labels(sid, [
            {"marker": "glucose", "pad_index": 0, "true_level": "negative"},
            {"marker": "blood",   "pad_index": 4, "true_level": "trace"},
        ])
        rows = db.export_dataset_csv(dsid)
        assert len(rows) == 2
        assert all("true_level" in r for r in rows)
        assert any(r["marker"] == "glucose" for r in rows)

    def test_group_split_no_overlap(self, tmp_path):
        """No sample appears in more than one split partition."""
        db = self._db(tmp_path)
        dsid = db.create_dataset("D8", "URS-4K")
        for i in range(20):
            db.add_sample(dsid, f"/tmp/img{i}.jpg", "URS-4K")
        counts = db.assign_group_split(dsid, train_frac=0.70, val_frac=0.15)
        assert counts["train"] + counts["val"] + counts["test"] == 20
        # Verify no duplicate splits — each sample must have exactly one split label
        samples = db.list_samples(dsid)
        splits = [s["split"] for s in samples]
        assert all(s in ("train", "val", "test") for s in splits), \
            f"Unexpected split values: {set(splits)}"

    def test_group_split_empty_dataset(self, tmp_path):
        """Empty dataset returns all-zero counts without error."""
        db = self._db(tmp_path)
        dsid = db.create_dataset("D9", "URS-4K")
        counts = db.assign_group_split(dsid)
        assert counts == {"train": 0, "val": 0, "test": 0}


class TestPhase4SyntheticGenerator:
    """Phase 4: synthetic image generator produces valid images with correct labels."""

    def test_generate_single_urs4k(self, tmp_path):
        from scripts.make_synthetic import generate_batch
        results = generate_batch(
            profile_id="URS-4K",
            count=1,
            output_dir=str(tmp_path),
            seed=42,
        )
        assert len(results) == 1
        r = results[0]
        assert os.path.isfile(r["image_path"])
        assert r["is_synthetic"] is True
        assert r["strip_type"] == "URS-4K"
        assert len(r["ground_truth"]) == 4  # one label per physical pad

    def test_generate_saliva4k(self, tmp_path):
        from scripts.make_synthetic import generate_batch
        results = generate_batch(
            profile_id="SALIVA-4K",
            count=3,
            output_dir=str(tmp_path),
            seed=7,
        )
        assert len(results) == 3
        for r in results:
            assert os.path.isfile(r["image_path"])

    def test_synthetic_image_readable_by_opencv(self, tmp_path):
        import cv2
        from scripts.make_synthetic import generate_batch
        results = generate_batch("URS-4K", 1, str(tmp_path), seed=0)
        img = cv2.imread(results[0]["image_path"])
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_ground_truth_level_in_chart(self, tmp_path):
        """Every generated pad level must exist in the reference chart."""
        from scripts.make_synthetic import generate_batch
        from app.ml.chart_colors import get_chart
        results = generate_batch("URS-4K", 2, str(tmp_path), seed=99)
        chart = get_chart("URS-4K")
        for r in results:
            for p in r["ground_truth"]:
                valid_levels = [lv for lv, _ in chart.get(p["marker"], [])]
                assert p["level"] in valid_levels, (
                    f"Level '{p['level']}' not in chart for marker '{p['marker']}'"
                )

    def test_synthetic_adds_to_db(self, tmp_path):
        """Synthetic samples registered in the DB have is_synthetic=1 and labels."""
        import sys
        sys.path.insert(0, str(tmp_path))  # ensure db accessible
        from scripts.make_synthetic import generate_batch, _add_to_db
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        dsid = db.create_dataset("SynthTest", "URS-4K")
        img_dir = str(tmp_path / "imgs")
        results = generate_batch("URS-4K", 2, img_dir, seed=5)
        _add_to_db(results, dsid, str(tmp_path / "test.db"))
        samples = db.list_samples(dsid)
        assert len(samples) == 2
        for s in samples:
            assert s["is_synthetic"] == 1
            labels = db.get_pad_labels(s["sample_id"])
            assert len(labels) == 4  # one label per physical pad

    def test_unknown_profile_raises(self, tmp_path):
        from scripts.make_synthetic import generate_batch
        with pytest.raises(ValueError, match="Unknown profile"):
            generate_batch("INVALID", 1, str(tmp_path))


# ===========================================================================
# Phase 5 — Training Engine
# ===========================================================================

class TestPhase5Features:
    """Phase 5: feature extraction shapes, white-reference differences, versioning."""

    def test_single_rgb_feature_shape(self):
        from app.ml.features import extract_pad_features, FEATURE_NAMES
        feat = extract_pad_features([200, 50, 80])
        assert feat.shape == (len(FEATURE_NAMES),), \
            f"Expected ({len(FEATURE_NAMES)},), got {feat.shape}"

    def test_patch_feature_shape(self):
        from app.ml.features import extract_pad_features, FEATURE_NAMES
        patch = np.ones((10, 10, 3), dtype=np.uint8) * 128
        feat = extract_pad_features(patch)
        assert feat.shape == (len(FEATURE_NAMES),)

    def test_white_diff_white_input(self):
        """A white pad should give near-zero Lab differences vs the white reference."""
        from app.ml.features import extract_pad_features
        feat = extract_pad_features([245, 245, 245], white_ref_rgb=[245, 245, 245])
        # Indices 17, 18, 19 are white_diff_L/a/b
        assert abs(feat[17]) < 2.0, f"white_diff_L too large: {feat[17]}"
        assert abs(feat[18]) < 2.0, f"white_diff_a too large: {feat[18]}"
        assert abs(feat[19]) < 2.0, f"white_diff_b too large: {feat[19]}"

    def test_white_delta_e_nonzero_for_colored_pad(self):
        from app.ml.features import extract_pad_features
        feat = extract_pad_features([255, 0, 0], white_ref_rgb=[245, 245, 245])
        # white_delta_e is index 20
        assert feat[20] > 10.0, f"Expected high delta_e for red vs white, got {feat[20]}"

    def test_feature_version_constant(self):
        from app.ml.features import FEATURE_VERSION
        assert FEATURE_VERSION == 1

    def test_batch_extraction_shape(self):
        from app.ml.features import extract_features_batch, FEATURE_NAMES
        rgbs = [[200, 50, 80], [100, 200, 30], [0, 0, 255]]
        feats = extract_features_batch(rgbs)
        assert feats.shape == (3, len(FEATURE_NAMES))


class TestPhase5ModelRegistry:
    """Phase 5: DB model registration, activation guard, rollback."""

    def _db(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        return DatabaseManager(str(tmp_path / "test.db"))

    def _fake_joblib_model(self, tmp_path, name="model.joblib"):
        import joblib as _jl
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        clf = Pipeline([("sc", StandardScaler()), ("clf", DummyClassifier())])
        clf.fit([[0, 0]], ["negative"])
        p = str(tmp_path / name)
        _jl.dump({"model": clf, "classes": ["negative", "+1"]}, p, compress=0)
        return p

    def _sha(self, path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def test_register_and_get_model(self, tmp_path):
        db = self._db(tmp_path)
        path = self._fake_joblib_model(tmp_path)
        sha = self._sha(path)
        mid = db.register_model(
            strip_type="URS-4K", marker="glucose",
            algorithm="rf", feature_version=1,
            metrics={"val": {"accuracy": 0.85, "macro_f1": 0.80}},
            model_path=path, sha256_hash=sha,
        )
        assert mid.startswith("MDL_")
        m = db.get_model(mid)
        assert m["strip_type"] == "URS-4K"
        assert m["version"] == 1

    def test_version_increments(self, tmp_path):
        db = self._db(tmp_path)
        for i in range(3):
            path = self._fake_joblib_model(tmp_path, f"m{i}.joblib")
            sha = self._sha(path)
            db.register_model(
                strip_type="URS-4K", marker="glucose",
                algorithm="rf", feature_version=1,
                metrics={}, model_path=path, sha256_hash=sha,
            )
        models = db.list_models(strip_type="URS-4K", marker="glucose")
        versions = sorted(m["version"] for m in models)
        assert versions == [1, 2, 3], f"Expected [1,2,3], got {versions}"

    def test_activate_deactivates_others(self, tmp_path):
        db = self._db(tmp_path)
        ids = []
        for i in range(2):
            path = self._fake_joblib_model(tmp_path, f"mod{i}.joblib")
            sha = self._sha(path)
            mid = db.register_model(
                strip_type="URS-4K", marker="glucose",
                algorithm="rf", feature_version=1,
                metrics={"accuracy": 0.9}, model_path=path, sha256_hash=sha,
            )
            ids.append(mid)
        db.activate_model(ids[0])
        db.activate_model(ids[1])
        active = db.get_active_model("URS-4K", "glucose")
        assert active["model_id"] == ids[1]
        m0 = db.get_model(ids[0])
        assert m0["is_active"] == 0

    def test_hash_verification_on_load(self, tmp_path):
        from app.ml.trainer import load_model
        import joblib as _jl
        path = self._fake_joblib_model(tmp_path, "hash_test.joblib")
        real_sha = self._sha(path)
        # Correct hash: should load fine
        artifact = load_model(path, real_sha)
        assert "model" in artifact
        # Wrong hash: should raise
        with pytest.raises(ValueError, match="hash mismatch"):
            load_model(path, "deadbeef" * 8)

    def test_delete_inactive_model(self, tmp_path):
        db = self._db(tmp_path)
        path = self._fake_joblib_model(tmp_path, "del.joblib")
        sha = self._sha(path)
        mid = db.register_model(
            strip_type="URS-4K", marker="protein",
            algorithm="gb", feature_version=1,
            metrics={}, model_path=path, sha256_hash=sha,
        )
        ok = db.delete_model(mid)
        assert ok
        assert db.get_model(mid) is None


class TestPhase5TrainingJobs:
    """Phase 5: training job lifecycle in the DB."""

    def _db(self, tmp_path):
        from app.database.db_manager import DatabaseManager
        return DatabaseManager(str(tmp_path / "jobs.db"))

    def test_create_and_poll_job(self, tmp_path):
        db = self._db(tmp_path)
        jid = db.create_training_job("URS-4K", "glucose", "rf")
        assert jid.startswith("JOB_")
        job = db.get_training_job(jid)
        assert job["status"] == "pending"
        assert job["progress"] == 0

    def test_only_one_training_job_can_be_active(self, tmp_path):
        from app.database.db_manager import TrainingCapacityError

        db = self._db(tmp_path)
        db.create_training_job("URS-4K", "glucose", "rf")
        with pytest.raises(TrainingCapacityError, match="already active"):
            db.create_training_job("URS-4K", "protein", "gb")

        active_job = db.list_active_training_jobs()[0]
        db.update_training_job(active_job["job_id"], status="cancelled")
        next_job = db.create_training_job("URS-4K", "protein", "gb")
        assert db.get_training_job(next_job)["status"] == "pending"

    def test_terminal_job_status_cannot_be_overwritten(self, tmp_path):
        db = self._db(tmp_path)
        job_id = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(job_id, status="cancelled")
        db.update_training_job(job_id, status="completed", model_id="MDL_LATE")
        job = db.get_training_job(job_id)
        assert job["status"] == "cancelled"
        assert job["model_id"] is None

    def test_update_job_status(self, tmp_path):
        db = self._db(tmp_path)
        jid = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(jid, status="running", progress=50, log_append="Half done.\n")
        job = db.get_training_job(jid)
        assert job["status"] == "running"
        assert job["progress"] == 50
        assert "Half done." in job["log_output"]

    def test_complete_job_records_model_id(self, tmp_path):
        db = self._db(tmp_path)
        jid = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(jid, status="completed", model_id="MDL_FAKE123")
        job = db.get_training_job(jid)
        assert job["status"] == "completed"
        assert job["model_id"] == "MDL_FAKE123"
        assert job["completed_at"] is not None

    def test_failed_job_records_error(self, tmp_path):
        db = self._db(tmp_path)
        jid = db.create_training_job("SALIVA-4K", "nitrite", "gb")
        db.update_training_job(jid, status="failed", error_message="Not enough data")
        job = db.get_training_job(jid)
        assert job["status"] == "failed"
        assert "Not enough data" in job["error_message"]

    def test_yolo_job_can_pause_and_resume(self, tmp_path):
        from flask import Flask
        from app.routes_training import training_bp

        db_path = str(tmp_path / "pause_jobs.db")
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path)
        job_id = db.create_training_job("URS-4K", "strip_detector", "yolov8_nano")
        db.update_training_job(job_id, status="running")

        flask_app = Flask(__name__)
        flask_app.config["DATABASE"] = db_path
        flask_app.register_blueprint(training_bp)
        client = flask_app.test_client()

        paused = client.post(f"/api/training/jobs/{job_id}/pause")
        assert paused.status_code == 200
        assert db.get_training_job(job_id)["status"] == "paused"

        resumed = client.post(f"/api/training/jobs/{job_id}/resume")
        assert resumed.status_code == 200
        assert db.get_training_job(job_id)["status"] == "running"

    def test_non_yolo_job_cannot_be_paused(self, tmp_path):
        from flask import Flask
        from app.routes_training import training_bp

        db_path = str(tmp_path / "non_yolo_pause_jobs.db")
        from app.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path)
        job_id = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(job_id, status="running")

        flask_app = Flask(__name__)
        flask_app.config["DATABASE"] = db_path
        flask_app.register_blueprint(training_bp)

        response = flask_app.test_client().post(f"/api/training/jobs/{job_id}/pause")
        assert response.status_code == 409
        assert db.get_training_job(job_id)["status"] == "running"

    def test_training_route_rejects_multiple_yolo_datasets(self, tmp_path, monkeypatch):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        import app.routes_training as routes_training

        db_path = str(tmp_path / "multiple_yolo.db")
        upload_dir = tmp_path / "uploads"
        db = DatabaseManager(db_path)
        datasets = [
            db.create_dataset("YOLO one", "URS-4K"),
            db.create_dataset("YOLO two", "URS-4K"),
        ]
        yolo_dir = upload_dir / "yolo_datasets"
        for dataset_id in datasets:
            dataset_dir = yolo_dir / dataset_id
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "data.yaml").write_text("train: train/images\nval: valid/images\n")

        flask_app = Flask(__name__)
        flask_app.config.update(DATABASE=db_path, UPLOAD_DIR=str(upload_dir))
        flask_app.register_blueprint(routes_training.training_bp)
        monkeypatch.setattr(
            routes_training.subprocess,
            "Popen",
            lambda *args, **kwargs: pytest.fail("Invalid YOLO request launched a process")
        )

        response = flask_app.test_client().post("/api/training/jobs", json={
            "strip_type": "URS-4K",
            "algorithm": "yolov8_nano",
            "dataset_ids": datasets,
        })
        assert response.status_code == 400
        assert "exactly one" in response.get_json()["error"]

    def test_yolo_training_launch_uses_selected_dataset_and_cpu_limits(
        self, tmp_path, monkeypatch
    ):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        import app.routes_training as routes_training

        db_path = str(tmp_path / "single_yolo.db")
        upload_dir = tmp_path / "uploads"
        db = DatabaseManager(db_path)
        dataset_id = db.create_dataset("YOLO dataset", "URS-4K")
        dataset_dir = upload_dir / "yolo_datasets" / dataset_id
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "data.yaml").write_text("train: train/images\nval: valid/images\n")

        flask_app = Flask(__name__)
        flask_app.config.update(
            DATABASE=db_path,
            UPLOAD_DIR=str(upload_dir),
            MODELS_DIR=str(tmp_path / "models"),
        )
        flask_app.register_blueprint(routes_training.training_bp)
        class FakeProcess:
            pid = 12345
            stdout = None

        captured = {}
        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr(routes_training.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(routes_training, "_monitor_process", lambda *args: None)
        routes_training._RUNNING_PROCESSES.clear()

        response = flask_app.test_client().post("/api/training/jobs", json={
            "strip_type": "URS-4K",
            "algorithm": "yolov8_nano",
            "dataset_ids": [dataset_id],
        })
        assert response.status_code == 202
        command = captured["command"]
        dataset_arg = command.index("--dataset-ids") + 1
        assert command[dataset_arg:] == [dataset_id]
        assert captured["kwargs"]["env"]["OMP_NUM_THREADS"] == "1"
        assert captured["kwargs"]["env"]["OPENBLAS_NUM_THREADS"] == "1"
        job = db.get_training_job(response.get_json()["job_id"])
        assert job["process_id"] == FakeProcess.pid
        routes_training._RUNNING_PROCESSES.clear()

    def test_job_poll_marks_exited_worker_failed(self, tmp_path, monkeypatch):
        from flask import Flask
        from app.database.db_manager import DatabaseManager
        import app.routes_training as routes_training

        db_path = str(tmp_path / "exited_worker.db")
        db = DatabaseManager(db_path)
        job_id = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(job_id, status="running", process_id=987654)

        flask_app = Flask(__name__)
        flask_app.config["DATABASE"] = db_path
        flask_app.register_blueprint(routes_training.training_bp)
        monkeypatch.setattr(routes_training, "_process_matches_job", lambda *args: False)

        response = flask_app.test_client().get(f"/api/training/jobs/{job_id}")
        job = response.get_json()
        assert response.status_code == 200
        assert job["status"] == "failed"
        assert "exited" in job["error_message"]

    def test_worker_monitor_captures_output_and_marks_unexpected_exit(self, tmp_path):
        from io import StringIO
        from app.database.db_manager import DatabaseManager
        from app.routes_training import _monitor_process

        db = DatabaseManager(str(tmp_path / "monitor_worker.db"))
        job_id = db.create_training_job("URS-4K", "glucose", "rf")
        db.update_training_job(job_id, status="running")

        class ExitedProcess:
            stdout = StringIO("worker diagnostic\n")

            def wait(self):
                return 1

        _monitor_process(job_id, ExitedProcess(), db)
        job = db.get_training_job(job_id)
        assert job["status"] == "failed"
        assert "worker diagnostic" in job["log_output"]
        assert "code 1" in job["error_message"]

    def test_yolo_loss_extraction_supports_component_mapping(self):
        from types import SimpleNamespace
        from app.ml.yolo_trainer import _training_loss

        trainer = SimpleNamespace(tloss={"box_loss": 0.25, "cls_loss": 0.5, "dfl_loss": 0.125})
        assert _training_loss(trainer) == 0.875

        trainer = SimpleNamespace(tloss=None, loss_items=[0.25, 0.5])
        assert _training_loss(trainer) == 0.75


class TestPhase5TrainEndToEnd:
    """Phase 5: fit->evaluate->save->load cycle on synthetic data."""

    def _make_labeled_db(self, tmp_path, db):
        """Create a dataset with enough synthetic samples (10 classes x 10 pads x 6 samples)."""
        import json as _json
        from app.ml.chart_colors import get_chart
        chart = get_chart("URS-4K")
        markers = list(chart.keys())
        marker = markers[0]
        levels = [lv for lv, _ in chart[marker]]

        dsid = db.create_dataset("TrainTest", "URS-4K")
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()

        rng = np.random.default_rng(42)
        for lvl in levels:
            for _ in range(6):
                # create a fake image file
                import cv2 as _cv2
                img = np.zeros((100, 30, 3), dtype=np.uint8)
                fname = str(img_dir / f"{uuid.uuid4().hex}.jpg")
                _cv2.imwrite(fname, img)
                sid = db.add_sample(dsid, fname, "URS-4K", is_synthetic=True)
                # Find color for this level in chart
                ref_color = next((c for lv, c in chart[marker] if lv == lvl), [128, 128, 128])
                pad_color = [min(255, max(0, c + int(rng.integers(-15, 15)))) for c in ref_color]
                db.save_pad_labels(sid, [{
                    "marker":       marker,
                    "pad_index":    0,
                    "true_level":   lvl,
                    "label_source": "chart",
                    "pad_color":    _json.dumps(pad_color),
                }])

        return dsid, marker

    def test_train_fit_predict_cycle(self, tmp_path):
        """End-to-end: fit RF, save, load (hash-verify), predict."""
        from app.database.db_manager import DatabaseManager
        from app.ml.trainer import train as do_train, load_model, predict
        from app.ml.features import extract_pad_features

        db_path = str(tmp_path / "e2e.db")
        db = DatabaseManager(db_path)
        dsid, marker = self._make_labeled_db(tmp_path, db)

        result = do_train(
            db_path=db_path,
            strip_type="URS-4K",
            marker=marker,
            algorithm="rf",
            dataset_ids=[dsid],
            n_estimators=20,
            seed=0,
            models_dir=str(tmp_path / "models"),
        )

        assert "model_id"   in result
        assert "model_path" in result
        assert "sha256"     in result

        # Verify hash-load
        artifact = load_model(result["model_path"], result["sha256"])
        assert "model" in artifact

        # Predict on a dummy feature
        feat = extract_pad_features([200, 50, 80])
        level, conf = predict(artifact, feat)
        assert isinstance(level, str)
        assert 0.0 <= conf <= 1.0

    def test_train_refuses_too_little_data(self, tmp_path):
        """Training with fewer than min_per_class samples per class must raise."""
        from app.database.db_manager import DatabaseManager
        from app.ml.trainer import train as do_train

        db_path = str(tmp_path / "tiny.db")
        db = DatabaseManager(db_path)
        dsid = db.create_dataset("Tiny", "URS-4K")

        # Add only 2 samples per class
        import json as _json
        levels = ["negative", "+1"]
        for lvl in levels:
            for _ in range(2):
                sid = db.add_sample(dsid, "/tmp/x.jpg", "URS-4K", is_synthetic=True)
                db.save_pad_labels(sid, [{
                    "marker": "glucose", "pad_index": 0,
                    "true_level": lvl, "label_source": "chart",
                    "pad_color": _json.dumps([128, 128, 128]),
                }])

        with pytest.raises(ValueError, match="Insufficient data"):
            do_train(
                db_path=db_path,
                strip_type="URS-4K", marker="glucose",
                algorithm="rf", dataset_ids=[dsid],
                n_estimators=10, min_per_class=5,
                models_dir=str(tmp_path / "models"),
            )

    def test_group_split_no_overlap(self, tmp_path):
        """No sample appears in both train and test partitions."""
        from app.ml.trainer import _group_split
        rng = np.random.default_rng(0)
        groups = np.array([f"photo_{i}" for i in range(30)])
        train_m, val_m, test_m = _group_split(groups)
        assert not np.any(train_m & test_m), "Train/test overlap!"
        assert not np.any(train_m & val_m),  "Train/val overlap!"
        assert not np.any(val_m  & test_m),  "Val/test overlap!"
        assert (train_m | val_m | test_m).all(), "Some samples not in any split!"


class TestSalivaPH4:
    def test_joint_match_handles_reversed_strip_orientation(self):
        from app.ml.chart_colors import SALIVA_PH4_PAD_CHART
        from app.ml.saliva_ph import predict_saliva_ph

        reference = SALIVA_PH4_PAD_CHART["7"]
        result = predict_saliva_ph(list(reversed(reference)))

        assert result["level"] == "7"
        assert result["reversed_chart_order"] is True
        assert len(result["pad_comparisons"]) == 4
        assert result["confidence"] > 0
        with pytest.raises(ValueError, match="exactly four"):
            predict_saliva_ph(reference[:3])

    def test_analyze_and_render_one_joint_ph_result(self, tmp_path, monkeypatch):
        from flask import Flask
        import cv2
        from app.database.db_manager import DatabaseManager
        from app.ml.chart_colors import SALIVA_PH4_PAD_CHART
        import app.routes as routes

        class FakeQualityChecker:
            def evaluate_image(self, image):
                return {
                    "retake_required": False,
                    "blur": 100.0,
                    "exposure": "good",
                    "glare": False,
                    "warnings": [],
                }

            def evaluate_strip_resolution(self, strip):
                return True, "ok"

        class FakeStripDetector:
            def __init__(self, yolo_detector=None):
                pass

            def detect(self, image, manual_corners=None):
                return {
                    "confidence": 1.0,
                    "method": "test",
                    "rectified_strip": np.full((100, 400, 3), 255, dtype=np.uint8),
                }

        class FakePadDetector:
            def detect_pads(self, image, strip, num_pads, **kwargs):
                assert num_pads == 4
                assert kwargs.get("marker_names") is None
                assert kwargs.get("detect_color_regions") is True
                return [
                    {"color": color, "color_raw": color, "index": index}
                    for index, color in enumerate(SALIVA_PH4_PAD_CHART["7"])
                ]

            def generate_debug_overlay(self, image, strip, pads):
                return image

        class FakeWhiteNormalizer:
            def normalize_strip_pads(self, strip, pads):
                return pads, {}

        monkeypatch.setattr(routes, "QualityGateChecker", FakeQualityChecker)
        monkeypatch.setattr(routes, "StripDetector", FakeStripDetector)
        monkeypatch.setattr(routes, "PadDetector", FakePadDetector)
        monkeypatch.setattr(routes, "WhiteReferenceNormalizer", FakeWhiteNormalizer)

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        image_path = upload_dir / "saliva-strip.png"
        assert cv2.imwrite(str(image_path), np.full((200, 400, 3), 255, dtype=np.uint8))

        app = Flask(
            "saliva_ph4_test",
            template_folder=os.path.join(ROOT, "app", "templates"),
        )
        app.config["UPLOAD_DIR"] = str(upload_dir)
        app.db = DatabaseManager(str(tmp_path / "saliva_ph4.db"))
        app.camera = None
        app.register_blueprint(routes.main_bp)
        patient_id = app.db.create_patient({"first_name": "Saliva", "last_name": "Test"})
        client = app.test_client()

        response = client.post("/api/analyze", json={
            "image_path": str(image_path),
            "patient_id": patient_id,
            "test_profile": "SALIVA-PH4",
        })

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert len(payload["results"]) == 1
        assert payload["results"][0]["marker"] == "pH"
        assert payload["results"][0]["result"] == "7"
        assert len(payload["results"][0]["pad_comparisons"]) == 4

        assessment = app.db.get_assessment(payload["assessment_id"])
        assert assessment["overall_status"] == "research"
        assert len(assessment["marker_results"][0]["pad_comparisons"]) == 4
        results_page = client.get(f"/results/{payload['assessment_id']}")
        assert results_page.status_code == 200
        assert b"Research-use color comparison only" in results_page.data
        assert b"<span class=\"quality-value\">Experimental</span>" in results_page.data
        assert b"Pad 4" in results_page.data
