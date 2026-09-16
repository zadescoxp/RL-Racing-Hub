import argparse
import csv
import os

import numpy as np

from environment import RacingEnv
from agents import (
    QLearningAgent,
    SarsaAgent,
    MonteCarloAgent,
    DQNAgent,
    PPOAgent,
)


def train_tabular(algo, episodes, out_dir):
    env = RacingEnv(max_steps=2500, training_laps=1)

    if algo == "qlearning":
        agent = QLearningAgent(env.n_actions)
    elif algo == "sarsa":
        agent = SarsaAgent(env.n_actions)
    else:
        agent = MonteCarloAgent(env.n_actions)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{algo}_training.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "completed_lap", "collisions", "epsilon"])

        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            collisions = 0
            done = False

            action = None
            if algo == "sarsa":
                action = agent.act(state, training=True)

            while not done:
                if algo == "sarsa":
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    next_action = (
                        agent.act(next_state, training=True)
                        if not done else 0
                    )
                    agent.learn(state, action, reward, next_state, next_action, done)
                    action = next_action
                else:
                    action = agent.act(state, training=True)
                    result = env.step(action)
                    next_state, reward, terminated, truncated, info = (
                        result.observation,
                        result.reward,
                        result.terminated,
                        result.truncated,
                        result.info,
                    )
                    done = terminated or truncated
                    agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                collisions += int(info["collision"])

            agent.end_episode()

            writer.writerow([
                ep,
                total_reward,
                env.step_count,
                int(env.laps >= 1),
                collisions,
                agent.epsilon,
            ])

            if ep == 1 or ep % 100 == 0:
                print(
                    f"[{algo}] episode={ep:5d} "
                    f"reward={total_reward:8.2f} "
                    f"lap={env.laps} "
                    f"collisions={collisions:3d} "
                    f"epsilon={agent.epsilon:.3f}"
                )

    agent.save(os.path.join(out_dir, f"{algo}.npy"))
    print(f"Saved {algo} -> {out_dir}/{algo}.npy")


def train_dqn(episodes, out_dir):
    env = RacingEnv(max_steps=2500, training_laps=1)
    agent = DQNAgent(len(env.get_observation()), env.n_actions)

    os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        collisions = 0

        while not done:
            action = agent.act(state, training=True)
            result = env.step(action)

            next_state = result.observation
            reward = result.reward
            done = result.terminated or result.truncated

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            collisions += int(result.info["collision"])

        agent.end_episode()

        if ep == 1 or ep % 25 == 0:
            print(
                f"[dqn] episode={ep:5d} "
                f"reward={total_reward:8.2f} "
                f"lap={env.laps} "
                f"collisions={collisions:3d} "
                f"epsilon={agent.epsilon:.3f}"
            )

    path = os.path.join(out_dir, "dqn.pt")
    agent.save(path)
    print(f"Saved DQN -> {path}")


def train_ppo(episodes, out_dir):
    env = RacingEnv(max_steps=2500, training_laps=1)
    agent = PPOAgent(len(env.get_observation()), env.n_actions)

    os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        collisions = 0

        while not done:
            action, log_prob, value = agent.act(state, training=True)
            result = env.step(action)

            next_state = result.observation
            reward = result.reward
            done = result.terminated or result.truncated

            agent.store(state, action, reward, done, log_prob, value)

            state = next_state
            total_reward += reward
            collisions += int(result.info["collision"])

            if len(agent.rollout) >= agent.rollout_size:
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = agent.act(state, training=False)
                agent.update(last_value)

        # Finish any partial rollout after the episode.
        if agent.rollout:
            agent.update(last_value=0.0 if done else agent.act(state, training=False)[2])

        if ep == 1 or ep % 10 == 0:
            print(
                f"[ppo] episode={ep:5d} "
                f"reward={total_reward:8.2f} "
                f"lap={env.laps} "
                f"collisions={collisions:3d}"
            )

    path = os.path.join(out_dir, "ppo.pt")
    agent.save(path)
    print(f"Saved PPO -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        required=True,
        choices=["qlearning", "sarsa", "montecarlo", "dqn", "ppo"],
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    np.random.seed(7)

    if args.algo in {"qlearning", "sarsa", "montecarlo"}:
        train_tabular(args.algo, args.episodes, args.out)
    elif args.algo == "dqn":
        train_dqn(args.episodes, args.out)
    else:
        train_ppo(args.episodes, args.out)


if __name__ == "__main__":
    main()
