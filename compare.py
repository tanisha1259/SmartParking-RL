import csv
import os
import pickle
import random

import matplotlib.pyplot as plt

from rl.q_learning import QLearningAgent
from sim.baseline_allocator import FirstAvailableAllocator
from sim.parking_env import ParkingEnvironment


TOTAL_SLOTS = 10
EPISODES = 100
POLICY_PATH = "models/policy_v1.pkl"
BEST_POLICY_PATH = "models/policy_best.pkl"
RESULTS_PATH = "results/comparison_results.csv"
SUMMARY_PATH = "results/evaluation_summary.txt"
PLOTS_DIR = "plots"


def run_episode(allocator, episode_seed):
    random.seed(episode_seed)

    env = ParkingEnvironment(total_slots=TOTAL_SLOTS)
    state = env.reset()

    done = False
    total_reward = 0
    total_wait_time = 0
    allocations = 0

    while not done:
        available_actions = env.available_slots()

        if isinstance(allocator, QLearningAgent):
            action = allocator.choose_action(state, available_actions)
        else:
            action = allocator.choose_action(available_actions)

        next_state, reward, done = env.park_car(action)

        total_reward += reward
        total_wait_time += env.last_wait_time
        allocations += 1
        state = next_state

    occupancy_rate = env.total_cars_parked / env.total_slots

    return {
        "reward": total_reward,
        "wait_time": total_wait_time / allocations,
        "occupancy_rate": occupancy_rate,
    }


def evaluate_allocator(name, allocator):
    rewards = []
    wait_times = []
    occupancy_rates = []

    for episode in range(EPISODES):
        metrics = run_episode(allocator, episode_seed=episode)
        rewards.append(metrics["reward"])
        wait_times.append(metrics["wait_time"])
        occupancy_rates.append(metrics["occupancy_rate"])

    return {
        "method": name,
        "average_reward": round(sum(rewards) / EPISODES, 2),
        "average_wait_time": round(sum(wait_times) / EPISODES, 2),
        "occupancy_rate": round(sum(occupancy_rates) / EPISODES, 2),
    }


def load_rl_agent():
    agent = QLearningAgent(num_actions=TOTAL_SLOTS, epsilon=0.0)
    policy_path = BEST_POLICY_PATH if os.path.exists(BEST_POLICY_PATH) else POLICY_PATH

    with open(policy_path, "rb") as file:
        agent.q_table = pickle.load(file)

    return agent


def save_bar_chart(results, metric, title, ylabel, output_path):
    methods = [result["method"] for result in results]
    values = [result[metric] for result in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, values, color=["#2f6f9f", "#7a7a7a"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_comparison_plots(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    save_bar_chart(
        results,
        "average_reward",
        "Average Reward Comparison",
        "Average Reward",
        os.path.join(PLOTS_DIR, "reward_comparison.png"),
    )
    save_bar_chart(
        results,
        "average_wait_time",
        "Average Wait Time Comparison",
        "Average Wait Time",
        os.path.join(PLOTS_DIR, "wait_time_comparison.png"),
    )
    save_bar_chart(
        results,
        "occupancy_rate",
        "Occupancy Rate Comparison",
        "Occupancy Rate",
        os.path.join(PLOTS_DIR, "occupancy_comparison.png"),
    )


def save_evaluation_summary(results):
    rl_metrics = results[0]
    baseline_metrics = results[1]

    reward_diff = rl_metrics["average_reward"] - baseline_metrics["average_reward"]
    wait_diff = rl_metrics["average_wait_time"] - baseline_metrics["average_wait_time"]
    occupancy_diff = rl_metrics["occupancy_rate"] - baseline_metrics["occupancy_rate"]

    if wait_diff < 0:
        wait_summary = "RL has lower average wait time than the baseline."
    elif wait_diff > 0:
        wait_summary = "Baseline has lower average wait time than RL."
    else:
        wait_summary = "RL and baseline have equal average wait time."

    with open(SUMMARY_PATH, "w") as file:
        file.write("SmartParking-RL Evaluation Summary\n")
        file.write("===================================\n\n")

        file.write("RL Allocation Metrics\n")
        file.write(f"- Average reward: {rl_metrics['average_reward']:.2f}\n")
        file.write(f"- Average wait time: {rl_metrics['average_wait_time']:.2f}\n")
        file.write(f"- Occupancy rate: {rl_metrics['occupancy_rate']:.2f}\n\n")

        file.write("Baseline First-Available Metrics\n")
        file.write(f"- Average reward: {baseline_metrics['average_reward']:.2f}\n")
        file.write(f"- Average wait time: {baseline_metrics['average_wait_time']:.2f}\n")
        file.write(f"- Occupancy rate: {baseline_metrics['occupancy_rate']:.2f}\n\n")

        file.write("Comparison\n")
        file.write(f"- Reward difference (RL - baseline): {reward_diff:.2f}\n")
        file.write(f"- Wait time difference (RL - baseline): {wait_diff:.2f}\n")
        file.write(f"- Occupancy difference (RL - baseline): {occupancy_diff:.2f}\n")
        file.write(f"- {wait_summary}\n")


def main():
    os.makedirs("results", exist_ok=True)

    rl_agent = load_rl_agent()
    baseline_allocator = FirstAvailableAllocator()

    results = [
        evaluate_allocator("RL allocation", rl_agent),
        evaluate_allocator("Baseline first-available", baseline_allocator),
    ]

    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "average_reward",
                "average_wait_time",
                "occupancy_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    save_comparison_plots(results)
    save_evaluation_summary(results)

    print(f"Comparison saved to {RESULTS_PATH}")
    print(f"Evaluation summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
