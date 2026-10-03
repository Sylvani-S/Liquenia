import cupy as cp
from .configs import CONFIG

def compute_gradient(f, kx, ky):
    """Numerical gradients in spectral space with exact periodicity."""
    f_hat = cp.fft.fft2(f)
    fx = cp.real(cp.fft.ifft2(1j * 2 * cp.pi * kx * f_hat))
    fy = cp.real(cp.fft.ifft2(1j * 2 * cp.pi * ky * f_hat))
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
        # Multi-channel / tensor field (e.g., shape (..., H, W))
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

def advect_maccormack(field, vx, vy, dt, grid_x, grid_y, W, H):
    """
    High-order MacCormack advection with RK2 trajectory tracing
    and monotonicity limiter clamping to prevent dissipation and overshoots.
    """
    # 1. Forward advection step (backward trace in time by dt)
    x_back, y_back = trace_rk2(grid_x, grid_y, vx, vy, dt, W, H)
    phi_star, phi_min, phi_max = sample_bilinear(field, x_back, y_back, W, H)

    # 2. Backward advection step (forward trace in time by dt)
    x_fwd, y_fwd = trace_rk2(grid_x, grid_y, vx, vy, -dt, W, H)
    phi_star_star, _, _ = sample_bilinear(phi_star, x_fwd, y_fwd, W, H)

    # 3. Error compensation and correction
    phi_corr = phi_star + 0.5 * (field - phi_star_star)

    # 4. Monotonicity limiter: clamp within local interpolation stencil
    return cp.clip(phi_corr, phi_min, phi_max)

def diffuse_and_project(v, viscosity, dt, kx, ky, k2, inv_k2):
    """
    Solves viscous diffusion and exact divergence-free pressure projection (Leray projection)
    simultaneously in Fourier space for incompressible Navier-Stokes.
    Supports shape (..., 2, H, W).
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

    return cp.real(cp.fft.ifft2(v_hat, axes=(-2, -1)))