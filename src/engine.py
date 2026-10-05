import cupy as cp
from .configs import CONFIG
from .math_utils import (
    compute_gradient,
    compute_laplacian,
    advect_maccormack,
    diffuse_and_project,
    _get_cuda_module,
)

class Tile:
    """Represents a spatial subdomain tile within the simulation grid."""
    def __init__(self, tx, ty, x0, y0, x1, y1, stream=None):
        self.tx = tx
        self.ty = ty
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.w = x1 - x0
        self.h = y1 - y0
        self.stream = stream or cp.cuda.Stream(non_blocking=True)


class TileGrid:
    """Manages 2D tile decomposition, bounding coordinates, and CUDA stream concurrency."""
    def __init__(self, W, H, tiles_x=4, tiles_y=4):
        self.W = W
        self.H = H
        self.tiles_x = max(1, int(tiles_x))
        self.tiles_y = max(1, int(tiles_y))
        self.tile_w = (W + self.tiles_x - 1) // self.tiles_x
        self.tile_h = (H + self.tiles_y - 1) // self.tiles_y
        self.tiles = []
        self.streams = []
        self._build_tiles()

    def _build_tiles(self):
        self.tiles = []
        self.streams = []
        for ty in range(self.tiles_y):
            tile_row = []
            stream_row = []
            for tx in range(self.tiles_x):
                x0 = tx * self.tile_w
                y0 = ty * self.tile_h
                x1 = min(self.W, (tx + 1) * self.tile_w)
                y1 = min(self.H, (ty + 1) * self.tile_h)
                stream = cp.cuda.Stream(non_blocking=True)
                tile = Tile(tx, ty, x0, y0, x1, y1, stream)
                tile_row.append(tile)
                stream_row.append(stream)
            self.tiles.append(tile_row)
            self.streams.append(stream_row)

    def synchronize(self):
        """Wait for all tile streams to complete."""
        cp.cuda.Device().synchronize()


class MultiStateSmoothLife:
    def __init__(self, W, H, S, dtype=CONFIG["dtype"], tile_grid=None):
        self.W, self.H, self.S = W, H, S
        self.dtype = dtype
        self.eps = CONFIG["eps"]

        # Configure 2D tile distribution
        if tile_grid is None:
            tile_grid = CONFIG.get("tile_grid", (4, 4))
        self.set_tile_grid(tile_grid[0], tile_grid[1])

        # Precompute coordinate grids for trajectory advection
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

        # Persistent advection buffers (zero reallocation in simulation loop)
        self._init_temp_buffers()

        self.randomize()
        self.new_board()
        self.kernel_fft = self._kernel_fft()

    def _init_temp_buffers(self):
        """Preallocate persistent temporary buffers for high-speed tiled advection."""
        self.v_temp_buffers = (
            cp.empty((2, self.H, self.W), dtype=self.dtype),
            cp.empty((2, self.H, self.W), dtype=self.dtype),
            cp.empty((2, self.H, self.W), dtype=self.dtype)
        )
        self.rho_temp_buffers = (
            cp.empty((self.H, self.W), dtype=self.dtype),
            cp.empty((self.H, self.W), dtype=self.dtype),
            cp.empty((self.H, self.W), dtype=self.dtype)
        )
        # Velocity self-advection cannot run in-place: the RK2 midpoint sampling
        # of every thread reads neighbouring cells of the very buffer that other
        # threads are writing, so it needs a separate destination.
        self.v_adv_out = cp.empty((2, self.H, self.W), dtype=self.dtype)

    def set_tile_grid(self, tiles_x, tiles_y):
        """Dynamically update tile distribution across the grid (e.g. 4x4, 8x8, 16x16)."""
        self.tiles_x = max(1, int(tiles_x))
        self.tiles_y = max(1, int(tiles_y))
        self.tile_grid_shape = (self.tiles_x, self.tiles_y)
        self.tile_manager = TileGrid(self.W, self.H, self.tiles_x, self.tiles_y)
        self.tile_streams = self.tile_manager.streams

    @property
    def tile_grid(self):
        """Current tile grid dimensions (tiles_x, tiles_y)."""
        return self.tile_grid_shape

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
        kernel = cp.exp(-r * 256.0)
        return cp.fft.fft2(kernel)

    def compute_potential(self):
        """Multi-species non-linear potential energy field."""
        weighted = cp.tensordot(self.B, self.rho, axes=(1, 0))
        f = cp.fft.fft2(weighted, axes=(1, 2))
        conv = cp.real(cp.fft.ifft2(f * self.kernel_fft, axes=(1, 2)))
        return cp.tensordot(self.A, conv, axes=(1, 0)) * self.mass[:, None, None]

    def _velocity_strides(self):
        """Element strides of the (S, 2, H, W) velocity stack, in elements."""
        plane = self.H * self.W
        return 2 * plane, plane

    def _ensure_v_contiguous(self):
        """
        Raw kernels index the velocity stack as a packed flat array, so it must
        be C-contiguous.  ``cp.real`` (and any external assignment) can leave it
        strided, which makes every tiled kernel read the wrong elements.
        """
        if not self.v.flags.c_contiguous:
            self.v = cp.ascontiguousarray(self.v)
        return self.v

    def _tile_velocity_clamp(self, max_v):
        """Apply maximum velocity clamping distributed across tiles."""
        mod = _get_cuda_module()
        is_f32 = (self.dtype == cp.float32)
        clamp_kernel = mod.get_function('velocity_clamp_tile_f32' if is_f32 else 'velocity_clamp_tile_f64')
        tw, th = self.tile_manager.tile_w, self.tile_manager.tile_h
        block = (16, 16)
        grid = ((tw + 15) // 16, (th + 15) // 16)
        state_stride, comp_stride = self._velocity_strides()

        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                with self.tile_streams[ty][tx]:
                    clamp_kernel(
                        grid, block,
                        (self._ensure_v_contiguous(), max_v,
                         self.W, self.H, self.S, state_stride, comp_stride,
                         tx * tw, ty * th, tw, th)
                    )
        self.tile_manager.synchronize()

    def _tile_add_body_force(self, gx, gy, dt):
        """Add potential gradient body forces distributed across tiles."""
        mod = _get_cuda_module()
        is_f32 = (self.dtype == cp.float32)
        force_kernel = mod.get_function('add_body_force_tile_f32' if is_f32 else 'add_body_force_tile_f64')
        tw, th = self.tile_manager.tile_w, self.tile_manager.tile_h
        block = (16, 16)
        grid = ((tw + 15) // 16, (th + 15) // 16)
        state_stride, comp_stride = self._velocity_strides()
        rho, gx, gy = self.rho, gx, gy
        if not rho.flags.c_contiguous:
            self.rho = rho = cp.ascontiguousarray(rho)
        if not gx.flags.c_contiguous:
            gx = cp.ascontiguousarray(gx)
        if not gy.flags.c_contiguous:
            gy = cp.ascontiguousarray(gy)

        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                with self.tile_streams[ty][tx]:
                    force_kernel(
                        grid, block,
                        (self._ensure_v_contiguous(), rho, gx, gy, dt,
                         self.W, self.H, self.S, state_stride, comp_stride,
                         tx * tw, ty * th, tw, th)
                    )
        self.tile_manager.synchronize()

    def step(self, dt=0.05, viscosity=0.001):
        """
        Execute one step of the simulation using incompressible Navier-Stokes
        with calculations distributed across 4x4 (or higher) tiles:
        1. Velocity clamping distributed across tiles.
        2. Non-linear velocity self-advection ((v · ∇)v) via tiled MacCormack with RK2.
        3. Force evaluation from potential gradients (-ρ · ∇Φ) distributed across tiles.
        4. Viscous diffusion and exact divergence-free Leray pressure projection (∇ · v = 0).
        5. Transport (advection) of species density fields distributed across tiles.
        6. Mass conservation normalization.
        """
        total_before = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        potential = self.compute_potential() * CONFIG["force_scale"]

        # Batched spectral gradient computation for potential field
        gx, gy = compute_gradient(potential, self.kx, self.ky)

        # Tile streams are created with non_blocking=True, so they do NOT
        # implicitly wait for the legacy default stream.  self.v was produced by
        # the previous step's FFT work on the default stream: publish it before
        # any tile kernel reads it.
        self.tile_manager.synchronize()

        # 1. Navier-Stokes Velocity Update: tile-distributed clamp
        self._tile_velocity_clamp(CONFIG["max_velocity"])

        # 2. Non-linear velocity self-advection distributed across tiles
        for s in range(self.S):
            advanced = advect_maccormack(
                self.v[s], self.v[s, 0], self.v[s, 1],
                dt, self.grid_x, self.grid_y, self.W, self.H,
                tile_grid=self.tile_grid_shape,
                streams=self.tile_streams,
                temp_buffers=self.v_temp_buffers,
                out=self.v_adv_out
            )
            self.v[s, 0] = advanced[0]
            self.v[s, 1] = advanced[1]

        # The copies above run on the default stream; the force kernels below run
        # on the tile streams and touch the same velocity buffer.
        self.tile_manager.synchronize()

        # 3. Body force from interaction potential distributed across tiles: f = -ρ · ∇Φ
        self._tile_add_body_force(gx, gy, dt)

        # 4. Viscous diffusion and divergence-free pressure projection in Fourier space
        self.v = diffuse_and_project(
            self.v, viscosity, dt, self.kx, self.ky, self.k2, self.inv_k2
        )

        # Same default-stream -> tile-stream handoff as above: the freshly
        # projected velocity lives on the default stream.
        self.tile_manager.synchronize()

        # 5. Density transport distributed across tiles
        for s in range(self.S):
            advect_maccormack(
                self.rho[s], self.v[s, 0], self.v[s, 1],
                dt, self.grid_x, self.grid_y, self.W, self.H,
                tile_grid=self.tile_grid_shape,
                streams=self.tile_streams,
                temp_buffers=self.rho_temp_buffers,
                out=self.rho[s]
            )

        # 6. Mass conservation normalization
        total_after = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        self.rho *= cp.where(total_before > self.eps, total_before / (total_after + self.eps), 0.0)

    def run(self, steps, dt, velocity):
        """Run multiple simulation sub-steps."""
        for _ in range(steps):
            self.step(dt, velocity)

    def randomize(self):
        """Randomize interaction parameters for multi-species dynamics."""
        self.A = cp.random.uniform(-16.0, 16.0, (self.S, self.S)).astype(self.dtype)
        self.B = cp.random.uniform(-16.0, 16.0, (self.S, self.S)).astype(self.dtype)
        self.mass = cp.random.uniform(1.0, 16.0, self.S, dtype=self.dtype)

    def new_board(self):
        """Reset board with random densities and zero velocities."""
        self.rho = cp.random.rand(self.S, self.H, self.W).astype(self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)

    def clear(self):
        """Clear density and velocity fields."""
        self.rho = cp.zeros((self.S, self.H, self.W), dtype=self.dtype)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=self.dtype)
