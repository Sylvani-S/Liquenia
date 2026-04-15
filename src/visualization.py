import pygame
import cupy as cp
import colorsys
import sys
from .configs import CONFIG

class Visualizer:
    def __init__(self, W, H, scale, states):
        self.W, self.H = W, H
        self.scale = scale
        self.states = states
        
        self.screen = pygame.display.set_mode((W * scale, H * scale))
        pygame.display.set_caption("CuPy Multi-State Smooth Life")
        self.screen_surf = pygame.Surface((W, H))
        
        state_colors = [colorsys.hsv_to_rgb(i / states, 0.8, 1.0) for i in range(states)]
        self.state_colors = cp.array(state_colors, dtype=cp.float64)

    def render(self, rho):
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
        rgb_final = cp.clip(rgb_final, 0, 255).astype(cp.uint8)

        # Blit to screen
        pygame.surfarray.blit_array(self.screen_surf, rgb_final.get().transpose(1, 0, 2))
        self.screen.blit(pygame.transform.scale(self.screen_surf, (self.W * self.scale, self.H * self.scale)), (0, 0))
        pygame.display.flip()

    def get_input(self):
        return pygame.event.get()

    def handle_mouse_drawing(self, sim):
        mx, my = pygame.mouse.get_pos()
        mx //= self.scale
        my //= self.scale
        r = 12
        x_start, x_end = max(0, mx - r), min(self.W, mx + r)
        y_start, y_end = max(0, my - r), min(self.H, my + r)
        sim.rho[:, y_start:y_end, x_start:x_end] = 1.0
