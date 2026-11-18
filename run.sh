#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "  LUMINE Health Screening Platform"
echo "  Starting up..."
echo "========================================"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

IS_RASPBERRY_PI=false
if { [ -r /proc/device-tree/model ] && tr -d '\000' < /proc/device-tree/model | grep -qi "Raspberry Pi"; } \
    || { [ -r /proc/cpuinfo ] && grep -qi "Raspberry Pi" /proc/cpuinfo; }; then
    IS_RASPBERRY_PI=true
fi

if [ "$IS_RASPBERRY_PI" = true ]; then
    echo "Running on Raspberry Pi"
    export LUMINE_MODE=production
    export DEMO_MODE=false
else
    echo "Running on development system"
    export LUMINE_MODE=development
    export DEMO_MODE=true
fi

VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ -e "$VENV_DIR" ]; then
        echo "The existing venv directory is incomplete or not a Linux virtual environment: $VENV_DIR" >&2
        exit 1
    fi

    if [ "$IS_RASPBERRY_PI" = true ]; then
        echo "Creating Raspberry Pi virtual environment..."
        python3 -m venv --system-site-packages "$VENV_DIR"
    else
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
fi

PYTHON="$VENV_DIR/bin/python"
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 12) else 1)'; then
    echo "The pinned dependencies require Python 3.9 through 3.11; found $("$PYTHON" --version)." >&2
    exit 1
fi

REQUIREMENTS_HASH="$(cksum < "$SCRIPT_DIR/requirements.txt")"
INSTALLED_HASH_FILE="$VENV_DIR/.requirements-hash"
INSTALLED_HASH=""
if [ -f "$INSTALLED_HASH_FILE" ]; then
    INSTALLED_HASH="$(cat "$INSTALLED_HASH_FILE")"
fi

if [ "$REQUIREMENTS_HASH" != "$INSTALLED_HASH" ]; then
    echo "Installing project dependencies..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    printf '%s\n' "$REQUIREMENTS_HASH" > "$INSTALLED_HASH_FILE"
fi

mkdir -p uploads calibration models logs

echo "Starting LUMINE..."
if "$PYTHON" app.py; then
    APP_STATUS=0
else
    APP_STATUS=$?
fi
echo "LUMINE has stopped."
if [ -t 0 ]; then
    read -r -p "Press Enter to exit..." || true
fi
exit "$APP_STATUS"
