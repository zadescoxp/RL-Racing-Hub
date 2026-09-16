import math
try:
    import pygame  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "pygame is required. Install it with: python -m pip install pygame"
    ) from exc

from environment import RacingEnv, TRACK_POINTS, TRACK_WIDTH
from agents import RandomAgent


WIDTH, HEIGHT = 1000, 850
SCALE = 1.0
FPS = 60


def world_to_screen(p):
    return (
        int(WIDTH / 2 + p[0] * SCALE),
        int(HEIGHT / 2 + p[1] * SCALE),
    )


def draw_track(screen):
    pts = [world_to_screen(p) for p in TRACK_POINTS]
    pygame.draw.lines(screen, (30, 120, 50), True, pts, int(TRACK_WIDTH + 10))
    pygame.draw.lines(screen, (150, 150, 150), True, pts, int(TRACK_WIDTH))
    pygame.draw.lines(screen, (240, 170, 50), True, pts, 3)

    for p in TRACK_POINTS:
        pygame.draw.circle(screen, (100, 100, 100), world_to_screen(p), 3)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Racing Lab - Environment Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    env = RacingEnv(max_steps=2500, training_laps=1)
    agent = RandomAgent(env.n_actions)

    state = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.act(state)
        result = env.step(action)
        state = result.observation

        if result.terminated or result.truncated:
            state = env.reset()

        screen.fill((245, 243, 238))
        draw_track(screen)

        p = world_to_screen(env.position)
        pygame.draw.circle(screen, (220, 50, 50), p, 9)

        text = (
            f"x={env.position[0]:.1f}  y={env.position[1]:.1f}  "
            f"vx={env.velocity_from_position[0]:.1f}  "
            f"vy={env.velocity_from_position[1]:.1f}  "
            f"speed={result.info['speed']:.1f}"
        )
        screen.blit(font.render(text, True, (20, 20, 20)), (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
