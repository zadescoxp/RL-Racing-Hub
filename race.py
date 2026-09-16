import os
import math
try:
    import pygame  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "pygame is required. Install it with: python -m pip install pygame"
    ) from exc

import numpy as np

from environment import RacingEnv, TRACK_POINTS, TRACK_WIDTH
from agents import (
    QLearningAgent,
    SarsaAgent,
    MonteCarloAgent,
    DQNAgent,
    PPOAgent,
)


WIDTH, HEIGHT = 1100, 900
PANEL_WIDTH = 330
SCALE = 0.95
FPS = 60
LAPS = 6


def world_to_screen(p):
    return (
        int((WIDTH - PANEL_WIDTH) / 2 + p[0] * SCALE),
        int(HEIGHT / 2 + p[1] * SCALE),
    )


def load_driver_image(path, size=48):
    image = pygame.transform.smoothscale(
        pygame.image.load(path).convert_alpha(),
        (size, size),
    )
    circle_mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(
        circle_mask,
        (255, 255, 255, 255),
        (size // 2, size // 2),
        size // 2,
    )
    image.blit(circle_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return image


class Racer:
    def __init__(self, name, algo, color, image_path):
        self.name = name
        self.model_name = {
            "qlearning": "Qlearn",
            "sarsa": "Sarsa",
            "montecarlo": "montecarlo",
            "dqn": "DQN",
            "ppo": "PPO",
        }[algo]
        self.algo = algo
        self.color = color
        self.image = load_driver_image(image_path)
        self.env = RacingEnv(max_steps=20000, training_laps=LAPS)
        self.state = self.env.reset()

        if algo == "qlearning":
            self.agent = QLearningAgent(self.env.n_actions)
            self.agent.epsilon = 0.0
            self.agent.q = np.load(
                "models/qlearning.npy",
                allow_pickle=True
            ).item()

        elif algo == "sarsa":
            self.agent = SarsaAgent(self.env.n_actions)
            self.agent.epsilon = 0.0
            self.agent.q = np.load(
                "models/sarsa.npy",
                allow_pickle=True
            ).item()

        elif algo == "montecarlo":
            self.agent = MonteCarloAgent(self.env.n_actions)
            self.agent.epsilon = 0.0
            self.agent.q = np.load(
                "models/montecarlo.npy",
                allow_pickle=True
            ).item()

        elif algo == "dqn":
            self.agent = DQNAgent(len(self.env.get_observation()), self.env.n_actions)
            self.agent.load("models/dqn.pt")

        elif algo == "ppo":
            self.agent = PPOAgent(len(self.env.get_observation()), self.env.n_actions)
            self.agent.load("models/ppo.pt")

        self.total_reward = 0.0
        self.finished = False
        self.time_steps = 0
        self.collisions = 0
        self.last_info = {
            "x": float(self.env.position[0]),
            "y": float(self.env.position[1]),
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "distance_from_centerline": 0.0,
            "progress": 0.0,
            "lap": 0,
            "checkpoint": 0,
        }
        self.dnf = False

    def step(self):
        if self.finished or self.dnf:
            return

        if self.algo == "ppo":
            action = self.agent.act(self.state, training=False)[0]
        else:
            action = self.agent.act(self.state, training=False)

        result = self.env.step(action)
        self.state = result.observation
        self.total_reward += result.reward
        self.time_steps += 1
        self.collisions += int(result.info["collision"])
        self.last_info = result.info

        if self.collisions > 100:
            self.dnf = True
            self.finished = True
        elif result.terminated or result.truncated:
            self.finished = True

    def status(self):
        if self.dnf:
            return "DNF"
        if self.env.laps >= LAPS:
            return "FINISHED"
        return f"Lap {self.env.laps}/{LAPS}"


def draw_track(screen):
    pts = [world_to_screen(p) for p in TRACK_POINTS]
    pygame.draw.lines(screen, (25, 120, 50), True, pts, int(TRACK_WIDTH + 12))
    pygame.draw.lines(screen, (145, 145, 145), True, pts, int(TRACK_WIDTH))
    pygame.draw.lines(screen, (240, 170, 50), True, pts, 3)


def main():
    global WIDTH, HEIGHT

    pygame.init()
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)

    available = [
        ("Charles LeCrash", "qlearning", (45, 110, 220), "models/qlearning.npy", "images/charles.jpeg"),
        ("Fernando Alon-slow", "sarsa", (245, 205, 35), "models/sarsa.npy", "images/alonso"),
        ("Lewis Hamiltone", "montecarlo", (125, 55, 185), "models/montecarlo.npy", "images/lewis.jpeg"),
        ("Max Verstoppin", "dqn", (225, 35, 35), "models/dqn.pt", "images/max.jpeg"),
        ("Lando No-Race", "ppo", (45, 175, 75), "models/ppo.pt", "images/lando.jpeg"),
    ]

    racers = []
    for name, algo, color, path, image_path in available:
        if os.path.exists(path):
            racers.append(Racer(name, algo, color, image_path))
        else:
            print(f"Skipping {name}: {path} not found.")

    if not racers:
        print("No trained models found. Train at least one agent first.")
        return

    pygame.display.set_caption("RL Racing Lab - Final Race")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    title_font = pygame.font.SysFont("Arial", 30, bold=True)
    panel_font = pygame.font.SysFont("Arial", 25, bold=True)
    model_font = pygame.font.SysFont("Arial", 16)
    telemetry_font = pygame.font.SysFont("Arial", 14)
    results_font = pygame.font.SysFont("Arial", 20, bold=True)

    running = True
    results_announced = False
    results_text = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        for racer in racers:
            racer.step()

        if not results_announced and all(racer.finished for racer in racers):
            results_announced = True
            ranked = sorted(
                racers,
                key=lambda racer: (racer.dnf, -racer.env.total_progress, racer.time_steps),
            )
            results_text = [
                f"{index}. {racer.name} - {racer.status()}"
                for index, racer in enumerate(ranked, start=1)
            ]
            print("\nFINAL RACE RESULTS")
            for result in results_text:
                print(result)

        screen.fill((0, 0, 0))
        draw_track(screen)

        panel_x = WIDTH - PANEL_WIDTH
        pygame.draw.rect(screen, (18, 18, 18), (panel_x, 0, PANEL_WIDTH, HEIGHT))
        pygame.draw.line(screen, (80, 80, 80), (panel_x, 0), (panel_x, HEIGHT), 2)

        screen.blit(
            title_font.render("RL Racing Lab - 6 Lap Evaluation", True, (245, 245, 245)),
            (25, 25),
        )

        screen.blit(
            panel_font.render("RACE DATA", True, (245, 245, 245)),
            (panel_x + 22, 28),
        )

        y = 78
        for racer in racers:
            p = world_to_screen(racer.env.position)
            screen.blit(racer.image, racer.image.get_rect(center=p))
            if racer.dnf:
                pygame.draw.circle(screen, (100, 100, 100), p, 25, 3)

            status = racer.status()
            info = racer.last_info

            screen.blit(font.render(racer.name, True, racer.color), (panel_x + 22, y))
            screen.blit(
                model_font.render(f"[{racer.model_name}]", True, (190, 190, 190)),
                (panel_x + 22, y + 20),
            )
            telemetry = [
                f"{status} | crashes {racer.collisions} | steps {racer.time_steps}",
                f"pos ({info['x']:6.1f}, {info['y']:6.1f})",
                f"vel ({info['vx']:5.1f}, {info['vy']:5.1f})  speed {info['speed']:5.1f}",
                f"center distance {info['distance_from_centerline']:5.1f}",
                f"progress {info['progress'] * 100:5.1f}%  lap {info['lap']}/{LAPS}",
                f"checkpoint {info['checkpoint']}  reward {racer.total_reward:7.1f}",
            ]
            for line_index, line in enumerate(telemetry):
                screen.blit(
                    telemetry_font.render(line, True, (205, 205, 205)),
                    (panel_x + 22, y + 37 + line_index * 13),
                )
            y += 125

        if results_announced:
            results_y = HEIGHT - 30 - len(results_text) * 22
            screen.blit(results_font.render("FINAL RESULTS", True, (245, 245, 245)), (25, results_y - 28))
            for index, result in enumerate(results_text):
                screen.blit(font.render(result, True, (225, 225, 225)), (25, results_y + index * 22))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
