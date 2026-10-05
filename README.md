
# Liquenia (Multi-State Smooth Life)

Implementation of **Smooth Life** with multiple interacting states, accelerated by CUDA via CuPy.

## Project Structure
- `main.py`: Entry point and application loop.
- `src/engine.py`: Core simulation logic and state management.
- `src/math_utils.py`: CUDA-accelerated math kernels.
- `src/configs.py`: Hyperparameters and global settings.
- `src/visualization.py`: Pygame rendering and input handling.

## Setup

1. **Automated Setup**:
   Simply run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Manual Installation**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage
```bash
source venv/bin/activate
python main.py
```

### Controls
- `R`: Randomize interaction matrices.
- `B`: Reset the board with random densities.
- `C`: Clear the board.
- `Z`: Increase total mass.
- `X`: Decrease total mass.
- `L`: Cycle through render modes.
- `Q`: Record/Stop recording.
- `UP/DOWN`: Increase/Decrease time step (`dt`).
- `LEFT/RIGHT`: Increase/Decrease `viscosity`.
- `+/-`: Increase/Decrease `frame_skip`.
- `T`: Cycle tile grid decomposition (4x4 -> 8x8 -> 16x16).
- `SHIFT`: Fine-tune parameter changes.
- `Mouse Click`: Draw density on the board.
