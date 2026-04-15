import cupy as cp
import numpy as np
import unittest
from src.engine import MultiStateSmoothLife
from src.configs import CONFIG

class TestConservation(unittest.TestCase):
    def setUp(self):
        self.W, self.H, self.S = 128, 128, 3
        self.sim = MultiStateSmoothLife(self.W, self.H, self.S)

    def test_mass_conservation(self):
        """Verify that total mass (sum of rho) is conserved after a step."""
        initial_mass = cp.sum(self.sim.rho).get()
        self.sim.step(dt=0.01, viscosity=0.1)
        final_mass = cp.sum(self.sim.rho).get()
        
        # Check if mass is conserved within float64 epsilon
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-10)

    def test_datatype(self):
        """Verify that the simulation is using the requested precision."""
        self.assertEqual(self.sim.rho.dtype, cp.float64)
        self.assertEqual(self.sim.v.dtype, cp.float64)

if __name__ == "__main__":
    unittest.main()
