import cupy as cp
from .configs import CONFIG
from .math_utils import compute_gradient, compute_laplacian, bilinear_advect

class MultiStateSmoothLife:
    def __init__(self, W, H, S, dtype=CONFIG["dtype"]):
        self.W, self.H, self.S = W, H, S
        self.dtype = dtype
        self.eps = CONFIG["eps"]

        self.rho = cp.zeros((S, H, W), dtype=self.dtype)
        self.v = cp.zeros((S, 2, H, W), dtype=self.dtype)
        
        self.randomize()
        self.new_board()
        self.kernel_fft = self._kernel_fft()
        
        ky, kx = cp.meshgrid(
            cp.fft.fftfreq(self.H),
            cp.fft.fftfreq(self.W),
            indexing='ij'
        )
        self.kx, self.ky = kx, ky

    def _kernel_fft(self):
        y, x = cp.meshgrid(
            cp.fft.fftfreq(self.H),
            cp.fft.fftfreq(self.W),
            indexing='ij'
        )
        r = cp.sqrt(x**2 + y**2)
        kernel = cp.exp(-r * 256.0)
        kernel_sum = cp.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        return cp.fft.fft2(kernel)

    def compute_potential(self):
        weighted = cp.tensordot(self.B, self.rho * self.mass[:, None, None], axes=(1, 0))
        f = cp.fft.fft2(weighted, axes=(1, 2))
        conv = cp.real(cp.fft.ifft2(f * self.kernel_fft, axes=(1, 2)))
        return cp.tensordot(self.A, cp.tanh(-conv), axes=(1, 0))

    def step(self, dt=0.05, viscosity=0.001):
        total_before = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        potential = self.compute_potential()
        
        for s in range(self.S):
            vx, vy = self.v[s]
            # 1. Advect
            self.rho[s] = bilinear_advect(self.rho[s], vx, vy, dt, self.W, self.H, self.dtype)
            # 2. Force
            gx, gy = compute_gradient(potential[s], self.kx, self.ky)
            self.v[s, 0] += -(gx * 512.0 + self.v[s, 0]) * dt
            self.v[s, 1] += -(gy * 512.0 + self.v[s, 1]) * dt
            # 3. Viscosity
            self.v[s, 0] += viscosity * compute_laplacian(self.v[s, 0])
            self.v[s, 1] += viscosity * compute_laplacian(self.v[s, 1])

        self.rho = cp.maximum(self.rho, self.eps)
        exponent = -0.5 / self.rho
        exponent = cp.clip(exponent, -50, 50)
        self.rho = cp.exp(exponent)

        total_after = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        self.rho *= total_before / (total_after + self.eps)

    def run(self, steps, dt, velocity):
        for _ in range(steps):
            self.step(dt, velocity)

    def randomize(self):
        self.A = cp.random.uniform(-100.0, 100.0, (self.S, self.S)).astype(self.dtype)
        self.B = cp.random.uniform(-100.0, 100.0, (self.S, self.S)).astype(self.dtype)
        self.mass = cp.random.uniform(0.1, 10.0, self.S, dtype=self.dtype)

    def new_board(self):
        self.rho = cp.random.rand(self.S, self.H, self.W).astype(self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)

    def clear(self):
        self.rho = cp.zeros((self.S, self.H, self.W), dtype=self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)
