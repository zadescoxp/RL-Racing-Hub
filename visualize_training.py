import argparse
import csv
import os
import math

try:
    import pygame  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "pygame is required. Install it with: python -m pip install pygame"
    ) from exc
import numpy as np

from environment import (
    RacingEnv,
    TRACK_POINTS,
    TRACK_WIDTH,
    TRACK_LENGTH,
)

from agents import (
    QLearningAgent,
    SarsaAgent,
    MonteCarloAgent,
    DQNAgent,
    PPOAgent,
)


# ============================================================
# CONFIG
# ============================================================

WIDTH = 1100
HEIGHT = 900

FPS = 60

BACKGROUND = (245, 243, 238)
TRACK_BORDER = (25, 120, 50)
TRACK_COLOR = (150, 150, 150)
CENTERLINE = (235, 170, 45)

TEXT = (25, 25, 25)
WHITE = (255, 255, 255)

AGENT_COLOR = (220, 50, 50)
CHECKPOINT_COLOR = (60, 100, 220)

SCALE = 1.15


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def world_to_screen(point):
    """
    Convert environment coordinates to pygame coordinates.
    """

    x = WIDTH / 2 + point[0] * SCALE
    y = HEIGHT / 2 + point[1] * SCALE

    return int(x), int(y)


# ============================================================
# DRAW TRACK
# ============================================================

def draw_track(screen):
    points = [
        world_to_screen(point)
        for point in TRACK_POINTS
    ]

    # Outer boundary
    pygame.draw.lines(
        screen,
        TRACK_BORDER,
        True,
        points,
        int((TRACK_WIDTH + 12) * SCALE),
    )

    # Road
    pygame.draw.lines(
        screen,
        TRACK_COLOR,
        True,
        points,
        int(TRACK_WIDTH * SCALE),
    )

    # Centerline
    pygame.draw.lines(
        screen,
        CENTERLINE,
        True,
        points,
        3,
    )

    # Track points
    for point in TRACK_POINTS:
        pygame.draw.circle(
            screen,
            (100, 100, 100),
            world_to_screen(point),
            3,
        )


# ============================================================
# DRAW CHECKPOINTS
# ============================================================

def draw_checkpoints(screen, env):
    for i, checkpoint in enumerate(env.checkpoints):

        point = TRACK_POINTS[int(checkpoint)]

        pygame.draw.circle(
            screen,
            CHECKPOINT_COLOR,
            world_to_screen(point),
            7,
            2,
        )


# ============================================================
# DRAW AGENT
# ============================================================

def draw_agent(screen, env):
    position = world_to_screen(env.position)

    # Main agent
    pygame.draw.circle(
        screen,
        AGENT_COLOR,
        position,
        10,
    )

    # Direction indicator
    direction_length = 22

    dx = math.cos(env.heading) * direction_length
    dy = math.sin(env.heading) * direction_length

    end = (
        int(position[0] + dx),
        int(position[1] + dy),
    )

    pygame.draw.line(
        screen,
        (20, 20, 20),
        position,
        end,
        3,
    )


# ============================================================
# UI
# ============================================================

def draw_text(screen, font, text, x, y):
    surface = font.render(text, True, TEXT)
    screen.blit(surface, (x, y))


def draw_panel(
    screen,
    font,
    small_font,
    algo,
    episode,
    total_reward,
    collisions,
    env,
    epsilon=None,
    fps_value=60,
):

    panel_x = 20
    panel_y = 20

    panel_width = 350
    panel_height = 300

    pygame.draw.rect(
        screen,
        WHITE,
        (
            panel_x,
            panel_y,
            panel_width,
            panel_height,
        ),
        border_radius=12,
    )

    title = f"RL Racing Lab — {algo.upper()}"

    draw_text(
        screen,
        font,
        title,
        panel_x + 18,
        panel_y + 18,
    )

    draw_text(
        screen,
        small_font,
        f"Episode: {episode}",
        panel_x + 18,
        panel_y + 60,
    )

    draw_text(
        screen,
        small_font,
        f"Reward: {total_reward:.2f}",
        panel_x + 18,
        panel_y + 85,
    )

    draw_text(
        screen,
        small_font,
        f"Lap: {env.laps}/{env.training_laps}",
        panel_x + 18,
        panel_y + 110,
    )

    draw_text(
        screen,
        small_font,
        f"Progress: {env.total_progress:.1f} / {TRACK_LENGTH:.1f}",
        panel_x + 18,
        panel_y + 135,
    )

    draw_text(
        screen,
        small_font,
        f"Speed: {env.speed:.2f}",
        panel_x + 18,
        panel_y + 160,
    )

    draw_text(
        screen,
        small_font,
        f"Collisions: {collisions}",
        panel_x + 18,
        panel_y + 185,
    )

    if epsilon is not None:
        draw_text(
            screen,
            small_font,
            f"Epsilon: {epsilon:.3f}",
            panel_x + 18,
            panel_y + 210,
        )

    draw_text(
        screen,
        small_font,
        f"Visualization FPS: {fps_value}",
        panel_x + 18,
        panel_y + 235,
    )

    draw_text(
        screen,
        small_font,
        "SPACE = pause    +/- = speed",
        panel_x + 18,
        panel_y + 265,
    )


# ============================================================
# CREATE AGENT
# ============================================================

def create_agent(algo, env):

    if algo == "qlearning":
        return QLearningAgent(env.n_actions)

    if algo == "sarsa":
        return SarsaAgent(env.n_actions)

    if algo == "montecarlo":
        return MonteCarloAgent(env.n_actions)

    if algo == "dqn":
        return DQNAgent(
            len(env.get_observation()),
            env.n_actions,
        )

    if algo == "ppo":
        return PPOAgent(
            len(env.get_observation()),
            env.n_actions,
        )

    raise ValueError(f"Unknown algorithm: {algo}")


# ============================================================
# TRAINING
# ============================================================

def train():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        required=True,
        choices=[
            "qlearning",
            "sarsa",
            "montecarlo",
            "dqn",
            "ppo",
        ],
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Number of environment steps per rendered frame.",
    )

    args = parser.parse_args()

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        f"RL Racing Lab — Training {args.algo.upper()}"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "Arial",
        22,
        bold=True,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        17,
    )

    env = RacingEnv(
        max_steps=2500,
        training_laps=1,
    )

    agent = create_agent(
        args.algo,
        env,
    )

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    csv_path = os.path.join(
        "logs",
        f"{args.algo}_visual_training.csv",
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
    )

    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "episode",
            "reward",
            "steps",
            "laps",
            "collisions",
            "speed",
        ]
    )

    paused = False

    simulation_speed = max(
        1,
        args.speed,
    )

    running = True

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(
        1,
        args.episodes + 1,
    ):

        state = env.reset()

        total_reward = 0.0
        collisions = 0

        done = False

        # SARSA needs an initial action
        action = None

        if args.algo == "sarsa":
            action = agent.act(
                state,
                training=True,
            )

        while not done and running:

            # ------------------------------------------------
            # EVENTS
            # ------------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key in (
                        pygame.K_EQUALS,
                        pygame.K_PLUS,
                    ):
                        simulation_speed = min(
                            simulation_speed + 1,
                            30,
                        )

                    elif event.key == pygame.K_MINUS:
                        simulation_speed = max(
                            simulation_speed - 1,
                            1,
                        )

            if not running:
                break

            if paused:

                screen.fill(BACKGROUND)

                draw_track(screen)
                draw_checkpoints(screen, env)
                draw_agent(screen, env)

                draw_panel(
                    screen,
                    font,
                    small_font,
                    args.algo,
                    episode,
                    total_reward,
                    collisions,
                    env,
                    getattr(agent, "epsilon", None),
                    FPS,
                )

                pygame.display.flip()

                clock.tick(30)

                continue

            # ------------------------------------------------
            # RUN MULTIPLE ENVIRONMENT STEPS
            # ------------------------------------------------

            for _ in range(simulation_speed):

                if done:
                    break

                # ============================================
                # SARSA
                # ============================================

                if args.algo == "sarsa":

                    result = env.step(action)

                    next_state = result.observation
                    reward = result.reward

                    done = (
                        result.terminated
                        or result.truncated
                    )

                    if not done:

                        next_action = agent.act(
                            next_state,
                            training=True,
                        )

                    else:

                        next_action = 0

                    agent.learn(
                        state,
                        action,
                        reward,
                        next_state,
                        next_action,
                        done,
                    )

                    action = next_action

                # ============================================
                # PPO
                # ============================================

                elif args.algo == "ppo":

                    action, log_prob, value = agent.act(
                        state,
                        training=True,
                    )

                    result = env.step(action)

                    next_state = result.observation
                    reward = result.reward

                    done = (
                        result.terminated
                        or result.truncated
                    )

                    agent.store(
                        state,
                        action,
                        reward,
                        done,
                        log_prob,
                        value,
                    )

                    if len(agent.rollout) >= agent.rollout_size:

                        if done:

                            last_value = 0.0

                        else:

                            _, _, last_value = agent.act(
                                next_state,
                                training=False,
                            )

                        agent.update(
                            last_value
                        )

                # ============================================
                # Q LEARNING / MONTE CARLO / DQN
                # ============================================

                else:

                    action = agent.act(
                        state,
                        training=True,
                    )

                    result = env.step(action)

                    next_state = result.observation
                    reward = result.reward

                    done = (
                        result.terminated
                        or result.truncated
                    )

                    agent.learn(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )

                state = next_state

                total_reward += reward

                collisions += int(
                    result.info["collision"]
                )

            # ------------------------------------------------
            # RENDER
            # ------------------------------------------------

            screen.fill(BACKGROUND)

            draw_track(screen)

            draw_checkpoints(
                screen,
                env,
            )

            draw_agent(
                screen,
                env,
            )

            draw_panel(
                screen,
                font,
                small_font,
                args.algo,
                episode,
                total_reward,
                collisions,
                env,
                getattr(agent, "epsilon", None),
                FPS,
            )

            pygame.display.flip()

            clock.tick(FPS)

        # ====================================================
        # END EPISODE
        # ====================================================

        if hasattr(agent, "end_episode"):
            agent.end_episode()

        # PPO partial rollout
        if args.algo == "ppo" and agent.rollout:

            agent.update(
                last_value=0.0
            )

        writer.writerow(
            [
                episode,
                total_reward,
                env.step_count,
                env.laps,
                collisions,
                env.speed,
            ]
        )

        csv_file.flush()

        if episode == 1 or episode % 25 == 0:

            print(
                f"[{args.algo}] "
                f"Episode {episode}/{args.episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Lap: {env.laps} | "
                f"Collisions: {collisions}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    csv_file.close()

    if args.algo in {
        "qlearning",
        "sarsa",
        "montecarlo",
    }:

        agent.save(
            f"models/{args.algo}.npy"
        )

    elif args.algo == "dqn":

        agent.save(
            "models/dqn.pt"
        )

    elif args.algo == "ppo":

        agent.save(
            "models/ppo.pt"
        )

    pygame.quit()

    print()
    print("Training complete.")
    print(f"Model saved in models/")
    print(f"Training log saved to {csv_path}")


if __name__ == "__main__":
    train()