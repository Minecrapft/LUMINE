#!/usr/bin/env python3
import os
import sys
import logging
from flask import Flask
from config import Config
from app.database.db_manager import DatabaseManager
from app.hardware.illumination import IlluminationController

# Try to import camera
try:
    from app.camera.camera_manager import CameraManager
    CAMERA_AVAILABLE = True
except Exception as e:
    CAMERA_AVAILABLE = False
    print(f"Camera module not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(config_overrides=None):
    app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)
    
    db = DatabaseManager(app.config['DATABASE_PATH'])
    app.db = db
    app.illumination = IlluminationController()
    app.config.setdefault('DATABASE',    app.config['DATABASE_PATH'])
    app.config.setdefault('MODELS_DIR',  os.path.join(os.path.dirname(__file__), 'models'))
    
    # Keep the camera closed until the capture page requests a preview.
    if CAMERA_AVAILABLE:
        try:
            app.camera = CameraManager()
        except Exception as e:
            logger.error(f"Camera error: {e}")
            app.camera = None
    else:
        logger.warning("Camera module not available")
        app.camera = None
    
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    from app.routes_dataset import dataset_bp
    app.register_blueprint(dataset_bp)

    from app.routes_training import training_bp
    app.register_blueprint(training_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    mode = os.environ.get('LUMINE_MODE', 'production')
    debug = mode == 'development'
    host = os.environ.get('LUMINE_HOST', '0.0.0.0')
    port = int(os.environ.get('LUMINE_PORT', 5000))
    logger.info(f"Starting LUMINE in {mode} mode on {host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
