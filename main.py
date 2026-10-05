import pygame
import sys
from src.configs import CONFIG
from src.engine import MultiStateSmoothLife
from src.visualization import Visualizer, VideoRecorder

def main():
    pygame.init()
    width, height = CONFIG["width"], CONFIG["height"]
    scale = CONFIG["scale"]
    states = CONFIG["states"]
    
    sim = MultiStateSmoothLife(width, height, states)
    viz = Visualizer(width, height, scale, states)
    recorder = VideoRecorder(width * scale, height * scale, CONFIG["target_fps"])
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
                    dt /= 2 ** (1 / 32 if fine else 1 / 8)
                    dt = max(dt, 0.00001)
                    print(f"DT: {dt:.4f}")
                elif event.key == pygame.K_UP:
                    dt *= 2 ** (1 / 32 if fine else 1 / 8)
                    dt = min(dt, 1.0)
                    print(f"DT: {dt:.4f}")
                elif event.key == pygame.K_a:
                    CONFIG["max_velocity"] *= 2 ** (0.125 if fine else 0.5)
                    print(f"Max Velocity: {CONFIG['max_velocity']:.4f}")
                elif event.key == pygame.K_s:
                    CONFIG["max_velocity"] /= 2 ** (0.125 if fine else 0.5)
                    print(f"Max Velocity: {CONFIG['max_velocity']:.4f}")
                elif event.key == pygame.K_LEFT:
                    viscosity /= 2 ** (1 / 32 if fine else 1 / 8)
                    viscosity = max(viscosity, 0)
                    print(f"Viscosity: {viscosity:.4f}")
                elif event.key == pygame.K_RIGHT:
                    viscosity *= 2 ** (1 / 32 if fine else 1 / 8)
                    viscosity = min(viscosity, 1.0)
                    print(f"Viscosity: {viscosity:.4f}")
                elif event.key == pygame.K_MINUS:
                    frame_skip = max(0, frame_skip - 1)
                    print(f"Frame Skip: {frame_skip}")
                elif event.key == pygame.K_EQUALS:
                    frame_skip = min(32, frame_skip + 1)
                    print(f"Frame Skip: {frame_skip}")
                elif event.key == pygame.K_z:
                    sim.total_mass = sim.total_mass * 2 ** (0.125 if fine else 0.5)
                    print(f"Total Mass: {sim.total_mass}")
                elif event.key == pygame.K_x:
                    sim.total_mass = sim.total_mass / 2 ** (0.125 if fine else 0.5)
                    print(f"Total Mass: {sim.total_mass}")
                elif event.key == pygame.K_COMMA:
                    CONFIG["force_scale"] *= 2 ** (0.125 if fine else 0.5)
                    print(f"Force Scale: {CONFIG['force_scale']:.4f}")
                elif event.key == pygame.K_PERIOD:
                    CONFIG["force_scale"] /= 2 ** (0.125 if fine else 0.5)
                    print(f"Force Scale: {CONFIG['force_scale']:.4f}")
                elif event.key == pygame.K_n:
                    CONFIG["mixing"] *= 2 ** (0.125 if fine else 0.5)
                    print(f"Mixing: {CONFIG['mixing']:.4f}")
                elif event.key == pygame.K_m:
                    CONFIG["mixing"] /= 2 ** (0.125 if fine else 0.5)
                    print(f"Mixing: {CONFIG['mixing']:.4f}")
                elif event.key == pygame.K_q:
                    if recorder.is_recording:
                        recorder.stop()
                    else:
                        recorder.start()
                elif event.key == pygame.K_l:
                    viz.cycle_mode()
                elif event.key == pygame.K_t:
                    tile_options = [(4, 4), (8, 8), (16, 16)]
                    curr_idx = tile_options.index(sim.tile_grid) if sim.tile_grid in tile_options else 0
                    next_tiles = tile_options[(curr_idx + 1) % len(tile_options)]
                    sim.set_tile_grid(next_tiles[0], next_tiles[1])
                    print(f"Switched tile grid: {next_tiles[0]}x{next_tiles[1]} tiles")
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    fine = False

        if dragging:
            viz.handle_mouse_drawing(sim)

        sim.run(frame_skip, dt, viscosity)
        viz.render(sim.rho, sim.v)
        recorder.write_frame(viz.screen)
        
        # Update title
        rec_status = "🔴 REC" if recorder.is_recording else "⚪ IDLE"
        mode_name = viz.MODE_NAMES[viz.render_mode]
        tiles_desc = f"{sim.tiles_x}x{sim.tiles_y} Tiles"
        title = f"Liquenia | {mode_name} | {tiles_desc} | {rec_status} | FPS: {clock.get_fps():.1f} | DT: {dt:.4f} | Visc: {viscosity:.4f} | Skip: {frame_skip}"
        pygame.display.set_caption(title)
        
        clock.tick(CONFIG["target_fps"])

    if recorder.is_recording:
        recorder.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
