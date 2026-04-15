import cupy as cp

# --- Configuration & Hyperparameters ---
CONFIG = {
    "width": 768,
    "height": 768,
    "scale": 1,
    "states": 3,
    "frame_skip": 1,
    "dt": 0.01,
    "viscosity": 0.33,
    "dtype": cp.float64,  # High precision
    "eps": 1e-12,         # Epsilon for float64
    "target_fps": 60
}
