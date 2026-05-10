import csv
import os

import matplotlib.pyplot as plt


REWARDS_PATH = "experiments/rewards.csv"
PLOTS_DIR = "plots"
OUTPUT_PATH = os.path.join(PLOTS_DIR, "reward_vs_episodes.png")


def load_rewards():
    episodes = []
    rewards = []

    with open(REWARDS_PATH, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            episodes.append(int(row["Episode"]))
            rewards.append(float(row["Reward"]))

    return episodes, rewards


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    episodes, rewards = load_rewards()

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes,
        rewards,
        color="#2f6f9f",
        linewidth=2,
        label="Episode reward",
    )
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Reward vs Episodes")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
