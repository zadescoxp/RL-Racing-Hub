import argparse
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    for path in args.files:
        rows = read_csv(path)
        rewards = [float(r["reward"]) for r in rows]
        episodes = [int(r["episode"]) for r in rows]
        label = path.split("/")[-1].replace("_training.csv", "")
        plt.plot(episodes, rewards, alpha=0.25, label=label)

        # Rolling mean.
        window = min(50, max(1, len(rewards) // 10))
        if len(rewards) >= window:
            rolling = [
                sum(rewards[i-window+1:i+1]) / window
                for i in range(window - 1, len(rewards))
            ]
            plt.plot(
                episodes[window - 1:],
                rolling,
                linewidth=2,
                label=f"{label} rolling mean",
            )

    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("RL Racing Training")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
