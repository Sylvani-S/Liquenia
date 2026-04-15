import cupy as cp
from .configs import CONFIG

def sigma(x, a, alpha):
    """Numerically stable sigmoid function."""
    alpha = max(alpha, CONFIG["eps"])
    # Clamp exponent to avoid overflow in exp
    exponent = -(x - a) * (4.0 / alpha)
    exponent = cp.clip(exponent, -50, 50) 
    return 1.0 / (1.0 + cp.exp(exponent))

def compute_gradient(f, kx, ky):
    """Numerical gradients in spectral space."""
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

def bilinear_advect(field, vx, vy, dt, W, H, dtype):
    """Semi-Lagrangian advection with bilinear interpolation."""
    y, x = cp.meshgrid(
        cp.arange(H, dtype=dtype),
        cp.arange(W, dtype=dtype),
        indexing='ij'
    )

    x_back = (x - dt * vx) % W
    y_back = (y - dt * vy) % H

    x0 = cp.floor(x_back).astype(cp.int32)
    y0 = cp.floor(y_back).astype(cp.int32)

    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H

    wx = x_back - x0
    wy = y_back - y0

    return (
        (1.0 - wx) * (1.0 - wy) * field[..., y0, x0] +
        wx * (1.0 - wy) * field[..., y0, x1] +
        (1.0 - wx) * wy * field[..., y1, x0] +
        wx * wy * field[..., y1, x1]
    )
