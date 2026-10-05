import unittest
import time
import cupy as cp
import numpy as np
from src.engine import MultiStateSmoothLife
from src.math_utils import sample_bilinear, trace_rk2

class TestTiling(unittest.TestCase):
    def setUp(self):
        self.W, self.H, self.S = 256, 256, 3
        self.sim = MultiStateSmoothLife(self.W, self.H, self.S, tile_grid=(4, 4))

    def test_tile_grid_initialization(self):
        """Verify default and custom tile grid properties."""
        self.assertEqual(self.sim.tile_grid, (4, 4))
        self.assertEqual(self.sim.tiles_x, 4)
        self.assertEqual(self.sim.tiles_y, 4)
        self.assertEqual(len(self.sim.tile_manager.tiles), 4)
        self.assertEqual(len(self.sim.tile_manager.tiles[0]), 4)

    def test_dynamic_tile_switching(self):
        """Verify dynamic reconfiguration between 4x4, 8x8, and 16x16 tiles."""
        for tx, ty in [(4, 4), (8, 8), (16, 16)]:
            self.sim.set_tile_grid(tx, ty)
            self.assertEqual(self.sim.tile_grid, (tx, ty))
            self.assertEqual(len(self.sim.tile_manager.tiles), ty)
            self.assertEqual(len(self.sim.tile_manager.tiles[0]), tx)
            # Ensure simulation can step normally after switching
            self.sim.step(dt=0.01, viscosity=0.05)

    def test_tiled_advection_consistency(self):
        """Verify that 4x4, 8x8, and 16x16 tile grids produce consistent results."""
        W, H = 128, 128
        sim4 = MultiStateSmoothLife(W, H, 1, tile_grid=(4, 4))
        sim8 = MultiStateSmoothLife(W, H, 1, tile_grid=(8, 8))

        # Synchronize states
        sim8.rho = sim4.rho.copy()
        sim8.v = sim4.v.copy()
        sim8.A = sim4.A.copy()
        sim8.B = sim4.B.copy()
        sim8.mass = sim4.mass.copy()

        sim4.step(dt=0.01, viscosity=0.05)
        sim8.step(dt=0.01, viscosity=0.05)

        max_rho_diff = float(cp.max(cp.abs(sim4.rho - sim8.rho)).get())
        max_v_diff = float(cp.max(cp.abs(sim4.v - sim8.v)).get())

        self.assertLess(max_rho_diff, 1e-10)
        self.assertLess(max_v_diff, 1e-10)

    def test_1024_real_time_performance(self):
        """Verify that 1024x1024 grid runs at real-time speeds (< 16.6ms per step = > 60 FPS)."""
        sim1024 = MultiStateSmoothLife(1024, 1024, 3, tile_grid=(4, 4))
        # Warmup
        for _ in range(3):
            sim1024.step(dt=0.01, viscosity=0.05)
        cp.cuda.Device().synchronize()

        t0 = time.time()
        N = 10
        for _ in range(N):
            sim1024.step(dt=0.01, viscosity=0.05)
        cp.cuda.Device().synchronize()
        t1 = time.time()

        step_time_ms = (t1 - t0) / N * 1000.0
        steps_per_sec = N / (t1 - t0)
        print(f"\n[Performance] 1024x1024 step time: {step_time_ms:.2f} ms ({steps_per_sec:.1f} steps/sec)")
        # Real-time requirement (<= 16.6 ms for 60 FPS)
        self.assertLess(step_time_ms, 16.66, f"Step time {step_time_ms:.2f}ms exceeds 16.66ms real-time budget")

if __name__ == "__main__":
    unittest.main()
