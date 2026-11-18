"""
LUMINE Configuration
"""

import logging
import os
import sys


def _read_bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, 'lumine.db')
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    CALIBRATION_DIR = os.path.join(BASE_DIR, 'calibration')
    
    for d in [UPLOAD_DIR, CALIBRATION_DIR]:
        os.makedirs(d, exist_ok=True)
    
    CAMERA_RESOLUTION = (1920, 1080)
    MIN_BLUR_THRESHOLD = 20.0
    MIN_BRIGHTNESS = 20.0
    MAX_BRIGHTNESS = 250.0
    MIN_CONTRAST = 5.0
    GLARE_THRESHOLD = 0.8
    
    # Sample types
    SAMPLE_TYPES = ['urine', 'saliva']
    DEFAULT_SAMPLE_TYPE = 'saliva'
    DEFAULT_TEST_PROFILE = 'SALIVA-4K'  # Changed to saliva
    
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DEMO_MODE = _read_bool_env("DEMO_MODE", default=True)
