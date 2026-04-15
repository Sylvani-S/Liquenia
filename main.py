import pygame
import sys
from src.configs import CONFIG
from src.engine import MultiStateSmoothLife
from src.visualization import Visualizer

def main():
    pygame.init()
    width, height = CONFIG["width"], CONFIG["height"]
    scale = CONFIG["scale"]
    states = CONFIG["states"]
    
    sim = MultiStateSmoothLife(width, height, states)
    viz = Visualizer(width, height, scale, states)
    clock = pygame.time.Clock()
    
    running = True
    fine = False
    dragging = False
    
    dt = CONFIG["dt"]
    viscosity = CONFIG["viscosity"]
    frame_skip = CONFIG["frame_skip"]

    while running:
        for event in viz.get_input():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    fine = True
                elif event.key == pygame.K_r:
                    sim.randomize()
                elif event.key == pygame.K_b:
                    sim.new_board()
                elif event.key == pygame.K_c:
                    sim.clear()
                elif event.key == pygame.K_DOWN:
                    dt -= 0.001 if fine else 0.01
                    dt = max(dt, 0)
                    print(f"DT: {dt:.4f}")
                elif event.key == pygame.K_UP:
                    dt += 0.001 if fine else 0.01
                    dt = min(dt, 1.0)
                    print(f"DT: {dt:.4f}")
                elif event.key == pygame.K_LEFT:
                    viscosity -= 0.001 if fine else 0.01
                    viscosity = max(viscosity, 0)
                    print(f"Viscosity: {viscosity:.4f}")
                elif event.key == pygame.K_RIGHT:
                    viscosity += 0.001 if fine else 0.01
                    viscosity = min(viscosity, 1.0)
                    print(f"Viscosity: {viscosity:.4f}")
                elif event.key == pygame.K_MINUS:
                    frame_skip = max(0, frame_skip - 1)
                    print(f"Frame Skip: {frame_skip}")
                elif event.key == pygame.K_EQUALS:
                    frame_skip = min(32, frame_skip + 1)
                    print(f"Frame Skip: {frame_skip}")
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    fine = False

        if dragging:
            viz.handle_mouse_drawing(sim)

        sim.run(frame_skip, dt, viscosity)
        viz.render(sim.rho)
        clock.tick(CONFIG["target_fps"])

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
