#!/bin/bash

# Configuration
VENV_DIR="venv"
PYTHON_BIN="python3"

echo "--- CuPy Multi-State Smooth Life Setup ---"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    $PYTHON_BIN -m venv $VENV_DIR
fi

# 2. Install dependencies
echo "[*] Installing/Updating dependencies..."
source $VENV_DIR/bin/activate
pip install -r requirements.txt

# 3. Verify CuPy installation
echo "[*] Verifying CuPy and CUDA..."
python -c "import cupy; print('CuPy version:', cupy.__version__); print('CUDA Device:', cupy.cuda.Device(0).attributes['MultiProcessorCount'], 'MPs')"

# 4. Instructions
echo "------------------------------------------"
echo "Setup Complete!"
echo "To activate the environment: source $VENV_DIR/bin/activate"
echo "To run the simulation:      python main.py"
echo "To run tests:               export PYTHONPATH=. && python tests/test_conservation.py"
echo "------------------------------------------"
