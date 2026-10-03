import cupy as cp
from .configs import CONFIG
from .math_utils import (
    compute_gradient,
    compute_laplacian,
    advect_maccormack,
    diffuse_and_project,
)

class MultiStateSmoothLife:
    def __init__(self, W, H, S, dtype=CONFIG["dtype"]):
        self.W, self.H, self.S = W, H, S
        self.dtype = dtype
        self.eps = CONFIG["eps"]

        # Precompute coordinate grids for fast trajectory advection
        self.grid_y, self.grid_x = cp.meshgrid(
            cp.arange(self.H, dtype=self.dtype),
            cp.arange(self.W, dtype=self.dtype),
            indexing='ij'
        )

        # Precompute spectral wavenumbers with Nyquist symmetry for derivative & projection accuracy
        kx = cp.fft.fftfreq(self.W)
        ky = cp.fft.fftfreq(self.H)
        kx[self.W // 2] = 0.0
        ky[self.H // 2] = 0.0
        self.ky, self.kx = cp.meshgrid(ky, kx, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2
        self.inv_k2 = cp.where(self.k2 > 0, 1.0 / self.k2, 0.0)

        # Initialize fields
        self.rho = cp.zeros((S, H, W), dtype=self.dtype)
        self.v = cp.zeros((S, 2, H, W), dtype=self.dtype)

        self.randomize()
        self.new_board()
        self.kernel_fft = self._kernel_fft()

    @property
    def total_mass(self):
        """Current total mass of the simulation."""
        return float(cp.sum(self.rho).get())

    @total_mass.setter
    def total_mass(self, new_mass):
        """Scale the density field to match a new total mass target."""
        curr = float(cp.sum(self.rho).get())
        if curr > self.eps:
            self.rho *= (new_mass / curr)

    def _kernel_fft(self):
        """Neighborhood interaction kernel normalized to unit sum."""
        y, x = cp.meshgrid(
            cp.fft.fftfreq(self.H),
            cp.fft.fftfreq(self.W),
            indexing='ij'
        )
        r = cp.sqrt(x**2 + y**2)
        kernel = cp.exp(-r * 128.0)
        kernel_sum = cp.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        return cp.fft.fft2(kernel)

    def compute_potential(self):
        """Multi-species non-linear potential energy field."""
        weighted = cp.tensordot(self.B, self.rho, axes=(1, 0))
        f = cp.fft.fft2(weighted, axes=(1, 2))
        conv = cp.real(cp.fft.ifft2(f * self.kernel_fft, axes=(1, 2)))
        return cp.tensordot(self.A, conv * self.mass[:, None, None], axes=(1, 0))

    def step(self, dt=0.05, viscosity=0.001):
        """
        Execute one step of the simulation using incompressible Navier-Stokes:
        1. Self-advection of velocity ((v · ∇)v) via high-order MacCormack with RK2 tracing.
        2. Force evaluation from multi-state interaction potential gradients (-ρ · ∇Φ).
        3. Viscous diffusion and exact divergence-free pressure projection (∇ · v = 0).
        4. Transport (advection) of species density fields by divergence-free flow.
        5. Cellular activation and exact mass conservation.
        """
        total_before = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        potential = self.compute_potential() * 32.0

        # --- Navier-Stokes Velocity Update ---
        v_norm = cp.sqrt(self.v[:, 0] ** 2 + self.v[:, 1] ** 2)
        self.v[:, 0] = cp.where(v_norm > CONFIG["max_velocity"], self.v[:, 0] * CONFIG["max_velocity"] / v_norm, self.v[:, 0])
        self.v[:, 1] = cp.where(v_norm > CONFIG["max_velocity"], self.v[:, 1] * CONFIG["max_velocity"] / v_norm, self.v[:, 1])
        
        for s in range(self.S):
            # 1. Non-linear velocity self-advection ((v · ∇)v)
            self.v[s] = advect_maccormack(
                self.v[s], self.v[s, 0], self.v[s, 1],
                dt, self.grid_x, self.grid_y, self.W, self.H
            )

            # 2. Body force from interaction potential: f = -ρ · ∇Φ
            gx, gy = compute_gradient(cp.sum(potential * self.mass[:, None, None], axis=0), self.kx, self.ky)
            self.v[s, 0] += -self.rho[s] * gx * dt
            self.v[s, 1] += -self.rho[s] * gy * dt

        # 3. Viscous diffusion and divergence-free pressure projection in Fourier space
        self.v = diffuse_and_project(
            self.v, viscosity, dt, self.kx, self.ky, self.k2, self.inv_k2
        )

        # --- Density Transport & Activation ---
        for s in range(self.S):
            # 4. Advect density with the divergence-free velocity field
            self.rho[s] = advect_maccormack(
                self.rho[s], self.v[s, 0], self.v[s, 1],
                dt, self.grid_x, self.grid_y, self.W, self.H
            )


        # 5. Mass conservation normalization
        total_after = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        self.rho *= cp.where(total_before > self.eps, total_before / (total_after + self.eps), 0.0)

    def run(self, steps, dt, velocity):
        """Run multiple simulation sub-steps."""
        for _ in range(steps):
            self.step(dt, velocity)

    def randomize(self):
        """Randomize interaction parameters for multi-species dynamics."""
        self.A = cp.random.uniform(-8.0, 8.0, (self.S, self.S)).astype(self.dtype)
        self.B = cp.random.uniform(-8.0, 8.0, (self.S, self.S)).astype(self.dtype)
        self.mass = cp.random.uniform(1.0, 4.0, self.S, dtype=self.dtype)

    def new_board(self):
        """Reset board with random densities and zero velocities."""
        self.rho = cp.random.rand(self.S, self.H, self.W).astype(self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)

    def clear(self):
        """Clear density and velocity fields."""
        self.rho = cp.zeros((self.S, self.H, self.W), dtype=self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)
