import pygame
import cupy as cp
import colorsys
import sys
import datetime
import cv2
from .configs import CONFIG
from .math_utils import compute_gradient

class Visualizer:
    def __init__(self, W, H, scale, states):
        self.W, self.H = W, H
        self.scale = scale
        self.states = states
        
        self.screen = pygame.display.set_mode((W * scale, H * scale))
        pygame.display.set_caption("Liquenia")
        self.screen_surf = pygame.Surface((W, H))
        
        state_colors = [colorsys.hsv_to_rgb(i / states, 0.8, 1.0) for i in range(states)]
        self.state_colors = cp.array(state_colors, dtype=cp.float64)

        self.render_mode = 0
        self.MODE_NAMES = ["Composite", "Edges", "Flex Fluid", "Velocity", "Surface Normal"]
        
        # Spectral utilities for gradients
        self.ky, self.kx = cp.meshgrid(
            cp.fft.fftfreq(H),
            cp.fft.fftfreq(W),
            indexing='ij'
        )

    def cycle_mode(self):
        self.render_mode = (self.render_mode + 1) % len(self.MODE_NAMES)
        print(f"Switched to mode: {self.MODE_NAMES[self.render_mode]}")

    def render(self, rho, v):
        rho2 = cp.where(rho > 0.01, rho * 0.9 + 0.1, rho)
        if self.render_mode == 0:
            self._render_composite(rho2)
        elif self.render_mode == 1:
            self._render_edges(rho2)
        elif self.render_mode == 2:
            self._render_flex(rho2)
        elif self.render_mode == 3:
            self._render_velocity(v)
        elif self.render_mode == 4:
            self._render_normals(rho2)

        # Blit to screen
        pygame.surfarray.blit_array(self.screen_surf, self.rgb_final.get().transpose(1, 0, 2))
        self.screen.blit(pygame.transform.scale(self.screen_surf, (self.W * self.scale, self.H * self.scale)), (0, 0))
        pygame.display.flip()

    def _render_composite(self, rho):
        # Normalize densities for display
        rho_min = cp.min(rho, axis=(1, 2), keepdims=True)
        rho_max = cp.max(rho, axis=(1, 2), keepdims=True)
        rho_norm = (rho - rho_min) / (rho_max - rho_min + CONFIG["eps"])

        # Composite RGB image
        rgb_accum = cp.zeros((self.H, self.W, 3), dtype=cp.float64)
        for i in range(self.states):
            layer_color = rho_norm[i, :, :, None] * self.state_colors[i]
            rgb_accum += layer_color ** 2

        rgb_final = cp.sqrt(rgb_accum) * 255.0
        self.rgb_final = cp.clip(rgb_final, 0, 255).astype(cp.uint8)

    def _render_edges(self, rho):
        total_rho = cp.sum(rho, axis=0)
        gx, gy = compute_gradient(total_rho, self.kx, self.ky)
        mag = cp.sqrt(gx**2 + gy**2)
        mag = (mag / (cp.max(mag) + CONFIG["eps"])) * 255.0
        
        rgb = cp.zeros((self.H, self.W, 3), dtype=cp.float64)
        rgb[:, :, 0] = mag * 0.5  # Cyan-ish edges
        rgb[:, :, 1] = mag
        rgb[:, :, 2] = mag
        self.rgb_final = cp.clip(rgb, 0, 255).astype(cp.uint8)

    def _render_flex(self, rho):
        total_rho = cp.sum(rho, axis=0)
        
        # Sharpen boundaries with sigmoid-like thresholding
        fluid_mask = 1.0 / (1.0 + cp.exp(-20.0 * (total_rho - 0.5)))
        
        # Calculate normals
        gx, gy = compute_gradient(fluid_mask, self.kx, self.ky)
        mag = cp.sqrt(gx**2 + gy**2 + 1e-6)
        nx, ny = -gx / mag, -gy / mag
        nz = cp.sqrt(cp.clip(1.0 - (nx**2 + ny**2), 0, 1))
        
        # Simple lighting
        lx, ly, lz = 0.577, 0.577, 0.577 # Normalized light dir [1,1,1]
        diffuse = cp.clip(nx * lx + ny * ly + nz * lz, 0, 1)
        
        # Specular (Blinn-Phong)
        hx, hy, hz = 0.0, 0.0, 1.0 # View direction (top down)
        half_x, half_y, half_z = (lx + hx)/2, (ly + hy)/2, (lz + hz)/2
        h_mag = cp.sqrt(half_x**2 + half_y**2 + half_z**2)
        half_x /= h_mag; half_y /= h_mag; half_z /= h_mag
        
        specular = cp.pow(cp.clip(nx * half_x + ny * half_y + nz * half_z, 0, 1), 32)
        
        # Fluid color (e.g., deep blue)
        base_color = cp.array([0.1, 0.4, 0.9])
        rgb = (base_color * diffuse[:, :, None] + specular[:, :, None]) * fluid_mask[:, :, None] * 255.0
        self.rgb_final = cp.clip(rgb, 0, 255).astype(cp.uint8)

    def _render_velocity(self, v):
        # Average velocity across states
        v_avg = cp.mean(v, axis=0)
        vx, vy = v_avg[0], v_avg[1]
        
        mag = cp.sqrt(vx**2 + vy**2)
        mag_norm = cp.clip(mag / (cp.max(mag) * 0.1 + CONFIG["eps"]), 0, 1)
        
        angle = (cp.get_array_module(vx).arctan2(vy, vx) + cp.pi) / (2 * cp.pi)
        
        # Convert HSV to RGB (Hue = angle, Sat = 1, Val = magnitude)
        # Using a simple manual HSV to RGB for CuPy
        h = angle * 6.0
        i = h.astype(cp.int32)
        f = h - i
        p = cp.full_like(h, 0.0)
        q = cp.full_like(h, 1.0 - f)
        t = f
        
        r = cp.choose(i % 6, cp.stack([cp.ones_like(h), q, p, p, t, cp.ones_like(h)]))
        g = cp.choose(i % 6, cp.stack([t, cp.ones_like(h), cp.ones_like(h), q, p, p]))
        b = cp.choose(i % 6, cp.stack([p, p, t, cp.ones_like(h), cp.ones_like(h), q]))
        
        rgb = cp.stack([r, g, b], axis=-1) * mag_norm[:, :, None] * 255.0
        self.rgb_final = cp.clip(rgb, 0, 255).astype(cp.uint8)

    def _render_normals(self, rho):
        total_rho = cp.sum(rho, axis=0)
        gx, gy = compute_gradient(total_rho, self.kx, self.ky)
        mag = cp.sqrt(gx**2 + gy**2 + 1e-6)
        
        # Normalize and map to 0-1 range for RGB
        nx = (gx / mag * 0.5 + 0.5)
        ny = (gy / mag * 0.5 + 0.5)
        nz = cp.ones_like(nx) * 0.5 + 0.5 # Flat blue-ish for Z
        
        rgb = cp.stack([nx, ny, nz], axis=-1) * 255.0
        # Only show where there is some density gradient
        mask = cp.clip(mag * 10.0, 0, 1)
        self.rgb_final = (rgb * mask[:, :, None]).astype(cp.uint8)

    def get_input(self):
        return pygame.event.get()

    def handle_mouse_drawing(self, sim):
        mx, my = pygame.mouse.get_pos()
        mx //= self.scale
        my //= self.scale
        r = 10
        x_start, x_end = max(0, mx - r), min(self.W, mx + r)
        y_start, y_end = max(0, my - r), min(self.H, my + r)
        sim.rho[:, y_start:y_end, x_start:x_end] += cp.random.rand(sim.S, y_end - y_start, x_end - x_start).astype(sim.dtype)
        sim.v[:, :, y_start:y_end, x_start:x_end] += cp.random.rand(sim.S, 2, y_end - y_start, x_end - x_start).astype(sim.dtype) * 4.0 - 2.0
class VideoRecorder:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.is_recording = False

    def start(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Liquenia_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
        self.is_recording = True
        print(f"Started recording to {filename}")

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        print("Stopped recording.")

    def write_frame(self, surface):
        if not self.is_recording or self.writer is None:
            return
        
        # Convert Pygame surface to OpenCV format (BGR)
        # pygame.surfarray.array3d returns (W, H, 3)
        # OpenCV needs (H, W, 3) and BGR
        frame = pygame.surfarray.array3d(surface)
        frame = frame.transpose(1, 0, 2)  # (H, W, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
