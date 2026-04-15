import numpy as np
import cupy as cp
import colorsys
import pygame
import random

width, height = 768,768
scale = 1
states = 3
frame_skip = 1
dt = 0.01
viscosity = 0.33
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((width * scale, height * scale))

def sigma(x, a, alpha):
    return 1 / (1 + cp.exp(-(x - a) * 4 / alpha))

class MultiStateSmoothLife:
    def __init__(self, W, H, S):
        self.W, self.H, self.S = W, H, S

        # --- densities (S, H, W)
        self.rho = cp.random.rand(S, H, W).astype(cp.float32)
        self.randomize()
        self.new_board()
        # --- kernel FFT
        self.kernel_fft = self._kernel_fft()


    def _kernel_fft(self):
        y, x = cp.meshgrid(
            cp.fft.fftfreq(self.H),
            cp.fft.fftfreq(self.W),
            indexing='ij'
        )
        r = cp.sqrt(x**2 + y**2)

        kernel = cp.exp(-r * 256)
        kernel /= cp.sum(kernel)

        return cp.fft.fft2(kernel)

    def gradient(self, f):
        ky, kx = cp.meshgrid(
            cp.fft.fftfreq(self.H),
            cp.fft.fftfreq(self.W),
            indexing='ij'
        )

        f_hat = cp.fft.fft2(f)

        fx = cp.real(cp.fft.ifft2(1j * kx * f_hat))
        fy = cp.real(cp.fft.ifft2(1j * ky * f_hat))

        return fx, fy
    
    def laplacian(self, f):
        return (
            4 * (
                cp.roll(f, 1, 0) + cp.roll(f, -1, 0) +
                cp.roll(f, 1, 1) + cp.roll(f, -1, 1)
            ) +
            (
                cp.roll(cp.roll(f, 1, 0), 1, 1) +
                cp.roll(cp.roll(f, 1, 0), -1, 1) +
                cp.roll(cp.roll(f, -1, 0), 1, 1) +
                cp.roll(cp.roll(f, -1, 0), -1, 1)
            ) -
            20 * f
        ) / 6.0

    # ----------------------------------
    # Semi-Lagrangian advection
    # ----------------------------------
    def advect(self, field, vx, vy, dt):
        y, x = cp.meshgrid(
            cp.arange(self.H),
            cp.arange(self.W),
            indexing='ij'
        )

        x_back = (x - dt * vx) % self.W
        y_back = (y - dt * vy) % self.H

        x0 = cp.floor(x_back).astype(cp.int32)
        y0 = cp.floor(y_back).astype(cp.int32)

        x1 = (x0 + 1) % self.W
        y1 = (y0 + 1) % self.H

        wx = x_back - x0
        wy = y_back - y0

        return (
            (1-wx)*(1-wy)*field[..., y0, x0] +
            wx*(1-wy)*field[..., y0, x1] +
            (1-wx)*wy*field[..., y1, x0] +
            wx*wy*field[..., y1, x1]
        )

    # ----------------------------------
    # Energy gradient (δE/δρ)
    # ----------------------------------
    def compute_potential(self):
        weighted = cp.tensordot(self.B, self.rho * self.mass[:, None, None], axes=(1, 0))

        f = cp.fft.fft2(weighted, axes=(1,2))
        conv = cp.real(cp.fft.ifft2(f * self.kernel_fft, axes=(1,2)))

        # δE/δρ_s
        return cp.tensordot(self.A,cp.tanh(-conv), axes=(1,0))

    # ----------------------------------
    # Step
    # ----------------------------------
    def step(self, dt=0.05, viscosity=0.001):

        total_before = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        # --- compute energy gradient
        potential = self.compute_potential()
        
        # --- update each state independently
        for s in range(self.S):
            vx, vy = self.v[s]

            # --- advection
            self.rho[s] = self.advect(self.rho[s], vx, vy, dt)


            # --- force = -grad(potential_s)
            gx, gy = self.gradient(potential[s])
            self.v[s,0] += -(gx * 512.0 + self.v[s,0]) * dt
            self.v[s,1] += -(gy * 512.0 + self.v[s,1]) * dt

            # --- viscosity
            self.v[s,0] += viscosity * self.laplacian(self.v[s,0])
            self.v[s,1] += viscosity * self.laplacian(self.v[s,1])
        self.rho = cp.exp(-.5 / self.rho)
        total_after = cp.sum(self.rho, axis=(1, 2), keepdims=True)
        self.rho *= total_before / (total_after + 1e-8)

    def run(self, steps, dt, velocity):
        for _ in range(steps):
            self.step(dt, velocity)
    def randomize(self):
        self.A = cp.random.uniform(-100.0, 100.0, (self.S, self.S)).astype(cp.float32)
        self.B = cp.random.uniform(-100.0, 100.0, (self.S, self.S)).astype(cp.float32)
        self.mass = cp.random.uniform(0.0, 10.0, self.S, dtype=cp.float32)
    def new_board(self):
        self.rho = cp.random.rand(self.S, self.H, self.W).astype(cp.float32)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=cp.float32)
    def clear(self):
        self.rho = cp.zeros_like(self.rho)
        self.v = cp.zeros((self.S, 2, self.H, self.W), dtype=cp.float32)


if __name__ == "__main__":
    sim = MultiStateSmoothLife(width, height, states)
    running = True
    screen2 = pygame.Surface((width, height))
    fine = False
    dragging = False
    color = [colorsys.hsv_to_rgb(i / states, 1, 1) for i in range(states)]
    color = cp.array(color)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.mod & pygame.KMOD_SHIFT:
                    fine = True
                if event.key == pygame.K_r:
                    sim.randomize()
                elif event.key == pygame.K_b:
                    sim.new_board()
                elif event.key == pygame.K_c:
                    sim.clear()
                elif event.key == pygame.K_DOWN:
                    dt -= 0.001 if fine else 0.01
                    dt = max(dt, 0)
                    print("DT: " + str(dt))
                elif event.key == pygame.K_UP:
                    dt += 0.001 if fine else 0.01
                    dt = min(dt, 1)
                    print("DT: " + str(dt))
                elif event.key == pygame.K_LEFT:
                    viscosity -= 0.001 if fine else 0.01
                    viscosity = max(viscosity, 0)
                    print("V: " + str(viscosity))
                elif event.key == pygame.K_RIGHT:
                    viscosity += 0.001 if fine else 0.01
                    viscosity = min(viscosity, 1)
                    print("V: " + str(viscosity))
                elif event.key == pygame.K_MINUS:
                    frame_skip = max(0, frame_skip - 1)
                    print("FS: " + str(frame_skip))
                elif event.key == pygame.K_EQUALS:
                    frame_skip = min(32, frame_skip + 1)
                    print("FS: " + str(frame_skip))
        if dragging:
            x, y = pygame.mouse.get_pos()
            x //= scale
            y //= scale
            if 0 <= x < width and 0 <= y < height:
                sim.rho[:, x-8:x+8, y-8:y+8] = 1.0
        fine = False
        sim.run(frame_skip, dt, viscosity)
        sim_grid = sim.rho
        sim_grid = (sim_grid - cp.min(sim_grid, axis=(1, 2))[:, None, None]) / (cp.max(sim_grid, axis=(1, 2))[:, None, None] - cp.min(sim_grid, axis=(1, 2))[:, None, None] + 1e-6)
        sim_rgb = cp.zeros((width, height, 3))
        for i in range(states):
            sim_rgb[:, :, :] += (sim_grid[i, cp.newaxis].transpose(1, 2, 0) * color[i, None, None] * 255) ** 2
        sim_rgb = cp.maximum(cp.minimum(cp.sqrt(sim_rgb), 255), 0)
        pygame.surfarray.blit_array(screen2, sim_rgb.astype(cp.uint8).get())
        screen.blit(pygame.transform.scale(screen2, (width * scale, height * scale)), (0, 0))
        pygame.display.flip()
        clock.tick(60)