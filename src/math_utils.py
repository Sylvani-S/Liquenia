import cupy as cp
from .configs import CONFIG

# --- CUDA RawModule for High-Performance Tiled Computations ---
_CUDA_TILED_SRC = r'''
template <typename T>
__device__ inline T sample_bilinear(const T* f, T px, T py, int W, int H) {
    int x0 = (int)floor(px);
    int y0 = (int)floor(py);
    T wx = px - (T)x0;
    T wy = py - (T)y0;
    x0 = (x0 % W + W) % W;
    y0 = (y0 % H + H) % H;
    int x1 = (x0 + 1) % W;
    int y1 = (y0 + 1) % H;
    T v00 = f[y0 * W + x0];
    T v10 = f[y0 * W + x1];
    T v01 = f[y1 * W + x0];
    T v11 = f[y1 * W + x1];
    return (1.0 - wx) * ((1.0 - wy) * v00 + wy * v01) + wx * ((1.0 - wy) * v10 + wy * v11);
}

template <typename T>
__device__ inline T sample_bilinear_minmax(
    const T* f, T px, T py, int W, int H,
    T& min_val, T& max_val
) {
    int x0 = (int)floor(px);
    int y0 = (int)floor(py);
    T wx = px - (T)x0;
    T wy = py - (T)y0;
    x0 = (x0 % W + W) % W;
    y0 = (y0 % H + H) % H;
    int x1 = (x0 + 1) % W;
    int y1 = (y0 + 1) % H;
    T v00 = f[y0 * W + x0];
    T v10 = f[y0 * W + x1];
    T v01 = f[y1 * W + x0];
    T v11 = f[y1 * W + x1];
    min_val = fmin(fmin(v00, v10), fmin(v01, v11));
    max_val = fmax(fmax(v00, v10), fmax(v01, v11));
    return (1.0 - wx) * ((1.0 - wy) * v00 + wy * v01) + wx * ((1.0 - wy) * v10 + wy * v11);
}

template <typename T>
__device__ inline void trace_rk2(
    T px, T py, const T* vx, const T* vy,
    T dt, int W, int H, T& out_x, T& out_y
) {
    int ix = ((int)px % W + W) % W;
    int iy = ((int)py % H + H) % H;
    T u = vx[iy * W + ix];
    T v = vy[iy * W + ix];
    T x_mid = px - (T)0.5 * dt * u;
    T y_mid = py - (T)0.5 * dt * v;
    T u_mid = sample_bilinear(vx, x_mid, y_mid, W, H);
    T v_mid = sample_bilinear(vy, x_mid, y_mid, W, H);
    out_x = px - dt * u_mid;
    out_y = py - dt * v_mid;
}

extern "C" {

__global__ void maccormack_tile_multi_pass1_f64(
    const double* __restrict__ field,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ phi_star,
    double* __restrict__ phi_min,
    double* __restrict__ phi_max,
    double dt, int W, int H, int C,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int plane_size = W * H;

    double xb, yb;
    trace_rk2((double)x, (double)y, vx, vy, dt, W, H, xb, yb);

    for (int c = 0; c < C; c++) {
        int c_offset = c * plane_size;
        double min_v, max_v;
        phi_star[c_offset + idx] = sample_bilinear_minmax(field + c_offset, xb, yb, W, H, min_v, max_v);
        phi_min[c_offset + idx] = min_v;
        phi_max[c_offset + idx] = max_v;
    }
}

// NOTE: 'out' intentionally has no __restrict__ qualifier: callers may pass
// 'out == field' for in-place advection. The load of field[idx] feeds the store
// to out[idx] within the same thread and must not be reordered or cached apart.
__global__ void maccormack_tile_multi_pass2_f64(
    const double* __restrict__ field,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ phi_star,
    const double* __restrict__ phi_min,
    const double* __restrict__ phi_max,
    double* out,
    double dt, int W, int H, int C,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int plane_size = W * H;

    double xf, yf;
    trace_rk2((double)x, (double)y, vx, vy, -dt, W, H, xf, yf);

    for (int c = 0; c < C; c++) {
        int c_offset = c * plane_size;
        double phi_star_star = sample_bilinear(phi_star + c_offset, xf, yf, W, H);
        double phi_corr = phi_star[c_offset + idx] + 0.5 * (field[c_offset + idx] - phi_star_star);
        out[c_offset + idx] = fmin(fmax(phi_corr, phi_min[c_offset + idx]), phi_max[c_offset + idx]);
    }
}

// The velocity stack is (S, 2, H, W). Raw kernels receive a single base
// pointer plus explicit strides instead of strided per-component views, so the
// caller never has to hand over a non-contiguous (stride 2*plane between
// states) slice.
__global__ void velocity_clamp_tile_f64(
    double* __restrict__ vel,
    double max_velocity,
    int W, int H, int S,
    long long state_stride, long long comp_stride,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    long long idx = (long long)y * W + x;

    for (int s = 0; s < S; s++) {
        long long base = (long long)s * state_stride + idx;
        double u = vel[base];
        double v_val = vel[base + comp_stride];
        double mag = sqrt(u * u + v_val * v_val);
        if (mag > max_velocity) {
            double scale = max_velocity / mag;
            vel[base] = u * scale;
            vel[base + comp_stride] = v_val * scale;
        }
    }
}

__global__ void add_body_force_tile_f64(
    double* __restrict__ vel,
    const double* __restrict__ rho,
    const double* __restrict__ gx,
    const double* __restrict__ gy,
    double dt,
    int W, int H, int S,
    long long state_stride, long long comp_stride,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    long long idx = (long long)y * W + x;

    for (int s = 0; s < S; s++) {
        long long base = (long long)s * state_stride + idx;
        long long field = (long long)s * W * H + idx;
        double fdt = -rho[field] * dt;
        vel[base] += fdt * gx[field];
        vel[base + comp_stride] += fdt * gy[field];
    }
}

__global__ void maccormack_tile_multi_pass1_f32(
    const float* __restrict__ field,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ phi_star,
    float* __restrict__ phi_min,
    float* __restrict__ phi_max,
    float dt, int W, int H, int C,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int plane_size = W * H;

    float xb, yb;
    trace_rk2((float)x, (float)y, vx, vy, dt, W, H, xb, yb);

    for (int c = 0; c < C; c++) {
        int c_offset = c * plane_size;
        float min_v, max_v;
        phi_star[c_offset + idx] = sample_bilinear_minmax(field + c_offset, xb, yb, W, H, min_v, max_v);
        phi_min[c_offset + idx] = min_v;
        phi_max[c_offset + idx] = max_v;
    }
}

// NOTE: 'out' intentionally has no __restrict__ qualifier: callers may pass
// 'out == field' for in-place advection.
__global__ void maccormack_tile_multi_pass2_f32(
    const float* __restrict__ field,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ phi_star,
    const float* __restrict__ phi_min,
    const float* __restrict__ phi_max,
    float* out,
    float dt, int W, int H, int C,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int plane_size = W * H;

    float xf, yf;
    trace_rk2((float)x, (float)y, vx, vy, -dt, W, H, xf, yf);

    for (int c = 0; c < C; c++) {
        int c_offset = c * plane_size;
        float phi_star_star = sample_bilinear(phi_star + c_offset, xf, yf, W, H);
        float phi_corr = phi_star[c_offset + idx] + 0.5f * (field[c_offset + idx] - phi_star_star);
        out[c_offset + idx] = fminf(fmaxf(phi_corr, phi_min[c_offset + idx]), phi_max[c_offset + idx]);
    }
}

__global__ void velocity_clamp_tile_f32(
    float* __restrict__ vel,
    float max_velocity,
    int W, int H, int S,
    long long state_stride, long long comp_stride,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    long long idx = (long long)y * W + x;

    for (int s = 0; s < S; s++) {
        long long base = (long long)s * state_stride + idx;
        float u = vel[base];
        float v_val = vel[base + comp_stride];
        float mag = sqrtf(u * u + v_val * v_val);
        if (mag > max_velocity) {
            float scale = max_velocity / mag;
            vel[base] = u * scale;
            vel[base + comp_stride] = v_val * scale;
        }
    }
}

__global__ void add_body_force_tile_f32(
    float* __restrict__ vel,
    const float* __restrict__ rho,
    const float* __restrict__ gx,
    const float* __restrict__ gy,
    float dt,
    int W, int H, int S,
    long long state_stride, long long comp_stride,
    int x0, int y0, int tile_w, int tile_h
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_w || ty >= tile_h) return;
    int x = x0 + tx;
    int y = y0 + ty;
    if (x >= W || y >= H) return;
    long long idx = (long long)y * W + x;

    for (int s = 0; s < S; s++) {
        long long base = (long long)s * state_stride + idx;
        long long field = (long long)s * W * H + idx;
        float fdt = -rho[field] * dt;
        vel[base] += fdt * gx[field];
        vel[base + comp_stride] += fdt * gy[field];
    }
}

}
'''

# Module cache
_MODULE = None

def _get_cuda_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(code=_CUDA_TILED_SRC)
    return _MODULE


def compute_gradient(f, kx, ky):
    """Numerical gradients in spectral space with exact periodicity.

    ``cp.real`` of a complex FFT result is a strided view that interleaves real
    and imaginary components, so the output is explicitly packed before being
    handed to consumers (notably the raw CUDA kernels, which index flat memory).
    """
    f_hat = cp.fft.fft2(f, axes=(-2, -1))
    fx = cp.ascontiguousarray(cp.real(cp.fft.ifft2(1j * 2 * cp.pi * kx * f_hat, axes=(-2, -1))))
    fy = cp.ascontiguousarray(cp.real(cp.fft.ifft2(1j * 2 * cp.pi * ky * f_hat, axes=(-2, -1))))
    return fx, fy


def compute_laplacian(f):
    """9-point Laplacian stencil for better isotropy."""
    return (
        4.0 * (
            cp.roll(f, 1, 0) + cp.roll(f, -1, 0) +
            cp.roll(f, 1, 1) + cp.roll(f, -1, 1)
        ) +
        (
            cp.roll(cp.roll(f, 1, 0), 1, 1) +
            cp.roll(cp.roll(f, 1, 0), -1, 1) +
            cp.roll(cp.roll(f, -1, 0), 1, 1) +
            cp.roll(cp.roll(f, -1, 0), -1, 1)
        ) -
        20.0 * f
    ) / 6.0


def sample_bilinear(f, px, py, W, H):
    """
    Sample a 2D or multi-channel periodic field with bilinear interpolation.
    Returns:
        interpolated_val, min_val, max_val
    where min_val and max_val represent the local 4-point stencil bounds
    used for monotonicity clamping in high-order advection.
    """
    x0 = cp.floor(px).astype(cp.int32) % W
    y0 = cp.floor(py).astype(cp.int32) % H
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H

    wx = px - cp.floor(px)
    wy = py - cp.floor(py)

    if f.ndim >= 3:
        wx = wx[None, ...]
        wy = wy[None, ...]
        v00 = f[..., y0, x0]
        v10 = f[..., y0, x1]
        v01 = f[..., y1, x0]
        v11 = f[..., y1, x1]
    else:
        v00 = f[y0, x0]
        v10 = f[y0, x1]
        v01 = f[y1, x0]
        v11 = f[y1, x1]

    val = (1.0 - wx) * ((1.0 - wy) * v00 + wy * v01) + wx * ((1.0 - wy) * v10 + wy * v11)
    min_val = cp.minimum(cp.minimum(v00, v10), cp.minimum(v01, v11))
    max_val = cp.maximum(cp.maximum(v00, v10), cp.maximum(v01, v11))
    return val, min_val, max_val


def trace_rk2(grid_x, grid_y, vx, vy, dt, W, H):
    """Runge-Kutta 2nd-order (Midpoint) trajectory tracing on a periodic domain."""
    x_mid = (grid_x - 0.5 * dt * vx) % W
    y_mid = (grid_y - 0.5 * dt * vy) % H
    vx_mid, _, _ = sample_bilinear(vx, x_mid, y_mid, W, H)
    vy_mid, _, _ = sample_bilinear(vy, x_mid, y_mid, W, H)
    x_back = (grid_x - dt * vx_mid) % W
    y_back = (grid_y - dt * vy_mid) % H
    return x_back, y_back


def tiled_maccormack_advect(
    field, vx, vy, dt, W, H,
    tile_grid=(4, 4),
    streams=None,
    temp_buffers=None,
    out=None
):
    """
    Distributed high-order MacCormack advection across NxN tiles on GPU.
    Supports single 2D fields (H, W) or multi-channel fields (C, H, W).

    'field', 'vx', 'vy' and 'out' must all be C-contiguous: the raw kernels
    index them as flat arrays.  'vx' and 'vy' must not alias any region of
    'out', because the RK2 midpoint sampling of neighbouring threads races with
    the in-flight writes of 'out'.
    """
    mod = _get_cuda_module()
    is_f32 = (field.dtype == cp.float32)
    pass1_kernel = mod.get_function('maccormack_tile_multi_pass1_f32' if is_f32 else 'maccormack_tile_multi_pass1_f64')
    pass2_kernel = mod.get_function('maccormack_tile_multi_pass2_f32' if is_f32 else 'maccormack_tile_multi_pass2_f64')

    tiles_x, tiles_y = tile_grid
    tile_w = (W + tiles_x - 1) // tiles_x
    tile_h = (H + tiles_y - 1) // tiles_y

    is_multi = (field.ndim >= 3)
    C = field.shape[0] if is_multi else 1

    if out is None:
        out = cp.empty_like(field)
    elif not out.flags.c_contiguous:
        raise ValueError("tiled_maccormack_advect: 'out' must be C-contiguous")

    for name, arr in (("field", field), ("vx", vx), ("vy", vy)):
        if not arr.flags.c_contiguous:
            raise ValueError(f"tiled_maccormack_advect: '{name}' must be C-contiguous")

    if temp_buffers is not None:
        phi_star, phi_min, phi_max = temp_buffers
    else:
        buf_shape = (C, H, W) if is_multi else (H, W)
        phi_star = cp.empty(buf_shape, dtype=field.dtype)
        phi_min = cp.empty(buf_shape, dtype=field.dtype)
        phi_max = cp.empty(buf_shape, dtype=field.dtype)

    block = (16, 16)
    grid = ((tile_w + 15) // 16, (tile_h + 15) // 16)

    # Phase 1: Forward advection and local stencil min/max
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            stream_ctx = streams[ty][tx] if streams else cp.cuda.Stream.null
            with stream_ctx:
                pass1_kernel(
                    grid, block,
                    (field, vx, vy, phi_star, phi_min, phi_max,
                     dt, W, H, C, tx * tile_w, ty * tile_h, tile_w, tile_h)
                )

    if streams:
        cp.cuda.Device().synchronize()

    # Phase 2: Backward error compensation and monotonicity limiter clamping
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            stream_ctx = streams[ty][tx] if streams else cp.cuda.Stream.null
            with stream_ctx:
                pass2_kernel(
                    grid, block,
                    (field, vx, vy, phi_star, phi_min, phi_max, out,
                     dt, W, H, C, tx * tile_w, ty * tile_h, tile_w, tile_h)
                )

    if streams:
        cp.cuda.Device().synchronize()

    return out


def advect_maccormack(field, vx, vy, dt, grid_x, grid_y, W, H, tile_grid=None, streams=None, temp_buffers=None, out=None):
    """
    High-order MacCormack advection with RK2 trajectory tracing
    and monotonicity limiter clamping distributed across tiles.
    """
    if tile_grid is None:
        tile_grid = CONFIG.get("tile_grid", (4, 4))

    return tiled_maccormack_advect(
        field, vx, vy, dt, W, H,
        tile_grid=tile_grid,
        streams=streams,
        temp_buffers=temp_buffers,
        out=out
    )


def diffuse_and_project(v, viscosity, dt, kx, ky, k2, inv_k2):
    """
    Solves viscous diffusion and exact divergence-free pressure projection (Leray projection)
    simultaneously in Fourier space for incompressible Navier-Stokes.
    Supports shape (..., 2, H, W).

    The result is explicitly made C-contiguous: ``cp.real`` on a complex array
    yields a strided view that interleaves real and imaginary components, which
    silently corrupts any raw kernel (or any integer-indexed consumer) that
    assumes a packed real layout.
    """
    v_hat = cp.fft.fft2(v, axes=(-2, -1))

    # Viscous diffusion (unconditionally stable exponential decay)
    if viscosity > 0:
        diffuse_decay = cp.exp(-viscosity * (4.0 * cp.pi**2 * k2) * dt)
        v_hat *= diffuse_decay

    # Leray projection: v_solenoidal = v - k * (k · v) / |k|^2
    dot = (kx * v_hat[..., 0, :, :] + ky * v_hat[..., 1, :, :]) * inv_k2
    v_hat[..., 0, :, :] -= kx * dot
    v_hat[..., 1, :, :] -= ky * dot

    return cp.ascontiguousarray(cp.real(cp.fft.ifft2(v_hat, axes=(-2, -1))))