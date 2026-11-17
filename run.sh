#!/bin/bash
# LUMINE Startup Script

echo "========================================"
echo "  LUMINE Health Screening Platform"
echo "  Starting up..."
echo "========================================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Running on Raspberry Pi - using production mode"
    export LUMINE_MODE=production
    export DEMO_MODE=True
else
    echo "Running on development system"
    export LUMINE_MODE=development
    export DEMO_MODE=True
fi

# Check for virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install/update dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Create necessary directories
mkdir -p uploads calibration models logs

# Start the application
echo "Starting LUMINE..."
python3 app.py

# If crashed, wait for user
echo "LUMINE has stopped."
read -p "Press Enter to exit..."