# LUMINE
## Intelligent Non-Invasive Health Screening & Assessment Platform

### Overview

LUMINE uses computer vision and colorimetric test-strip analysis to convert biological-sample test reactions into structured health-marker observations, identify possible health-condition associations, monitor trends over time, and recommend appropriate confirmatory evaluation.

**Core principle:** LUMINE performs preliminary health-marker screening and assessment, not diagnosis.

### Features

- **1080p Camera Capture** with automatic quality checking
- **Strip Detection** using YOLO + OpenCV
- **Color Analysis** with perceptual matching
- **Marker Interpretation** across multiple health domains
- **Condition Association** with evidence-level transparency
- **Patient Management** with history and trends
- **Research Mode** for experimental assays
- **Validation Mode** for ground-truth testing
- **Offline-First** - works without internet

### Health Domains

- Metabolic
- Renal / Urinary
- Hepatic
- Hydration
- Inflammation
- Acid/Base
- Oncology Research (experimental)

### Hardware Requirements

- Raspberry Pi 5
- 7-inch Touchscreen
- 1080p Camera (Raspberry Pi Camera Module or USB)
- JUSTO MOSFET Driver Module and 3 × 5V white LEDs
- Raspberry Pi BCM GPIO 18 (physical pin 12) for PWM brightness control
- Strip Holder
- Power Supply

For LED control, install the Raspberry Pi OS `lgpio` package
(`sudo apt install python3-lgpio`). If using a virtual environment, create it
with `python3 -m venv --system-site-packages venv` so it can import the OS
package, then install the project requirements. Run the Flask app on the Pi;
the browser can connect from another device on the network. Connect GPIO 18 to
the MOSFET module's PWM/input terminal and confirm the module accepts 3.3V logic.
Power the LEDs from a separate 5V supply through the MOSFET; never connect the
LED load directly to a Pi GPIO pin. Connect Pi ground to the module input ground
when required by the module's wiring.

### Software Requirements

- Python 3.9+
- Flask
- OpenCV
- YOLOv8
- SQLite3

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Caisio/LUMINE.git
cd LUMINE