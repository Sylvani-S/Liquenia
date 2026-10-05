import cupy as cp

# --- Configuration & Hyperparameters ---
CONFIG = {
    "width": 1024,
    "height": 1024,
    "scale": 1,
    "tile_grid": (4, 4),  # 4x4 tiles (distributes calculations across 16 tiles)
    "states": 3,
    "frame_skip": 1,
    "dt": 0.005,
    "viscosity": 0.005,
    "dtype": cp.float64,  # High precision
    "eps": 1e-12,         # Epsilon for float64
    "target_fps": 60,
    "max_velocity": 8192.0,
    "mixing": 1.0,
    "force_scale": 16.0
}
