import unittest
import cupy as cp
import numpy as np
from src.engine import MultiStateSmoothLife
from src.math_utils import compute_gradient

class TestNavierStokes(unittest.TestCase):
    def setUp(self):
        self.W, self.H, self.S = 128, 128, 3
        self.sim = MultiStateSmoothLife(self.W, self.H, self.S)

    def test_divergence_free_projection(self):
        """Verify that the velocity field satisfies incompressibility (∇ · v ≈ 0)."""
        self.sim.step(dt=0.05, viscosity=0.01)
        for s in range(self.S):
            div = cp.real(cp.fft.ifft2(
                1j * 2 * cp.pi * (
                    self.sim.kx * cp.fft.fft2(self.sim.v[s, 0]) +
                    self.sim.ky * cp.fft.fft2(self.sim.v[s, 1])
                )
            ))
            max_div = float(cp.max(cp.abs(div)).get())
            # Divergence should be close to machine precision for float64
            self.assertLess(max_div, 1e-10, f"State {s} has non-zero divergence: {max_div}")

    def test_density_non_negativity(self):
        """Verify that advection with monotonicity limiter maintains non-negative densities."""
        for _ in range(5):
            self.sim.step(dt=0.05, viscosity=0.01)
        min_rho = float(cp.min(self.sim.rho).get())
        self.assertGreaterEqual(min_rho, 0.0, "Density became negative after advection")

    def test_velocity_evolution(self):
        """Verify that velocity field develops non-zero physical circulation from potential."""
        self.sim.step(dt=0.05, viscosity=0.001)
        v_speed = cp.sqrt(self.sim.v[:, 0]**2 + self.sim.v[:, 1]**2)
        mean_speed = float(cp.mean(v_speed).get())
        self.assertGreater(mean_speed, 1e-6, "Velocity did not develop from forces")

if __name__ == "__main__":
    unittest.main()
