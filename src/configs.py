import cupy as cp

# --- Configuration & Hyperparameters ---
CONFIG = {
    "width": 512,
    "height": 512,
    "scale": 2,
    "states": 3,
    "frame_skip": 1,
    "dt": 0.05,
    "viscosity": 0.01,
    "dtype": cp.float64,  # High precision
    "eps": 1e-12,         # Epsilon for float64
    "target_fps": 60,
    "max_velocity": 16.0
}
