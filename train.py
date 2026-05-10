from sim.parking_env import ParkingEnvironment
from rl.q_learning import QLearningAgent

import matplotlib.pyplot as plt
import csv
import os
import pickle
import copy
import json
import random

# Create folders if not present
os.makedirs("experiments", exist_ok=True)
os.makedirs("models", exist_ok=True)

random.seed(42)

# Initialize environment
env = ParkingEnvironment(total_slots=10)

# Initialize agent
agent = QLearningAgent(
    num_actions=10,
    discount_factor=0.0,
    epsilon_decay=0.997,
    epsilon_min=0.05
)

# Training settings
episodes = 5000

# Store rewards
episode_rewards = []
episode_wait_times = []
episode_occupancy_rates = []
best_reward = None
best_q_table = None


for episode in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0
    total_wait_time = 0
    allocations = 0

    while not done:

        # Choose action
        available_actions = env.available_slots()

        action = agent.choose_action(
            state,
            available_actions
        )

        # Take action
        next_state, reward, done = env.park_car(action)

        # Update Q-table
        agent.update_q_table(
            state,
            action,
            reward,
            next_state,
            env.available_slots(),
            done
        )

        state = next_state

        total_reward += reward
        total_wait_time += env.last_wait_time
        allocations += 1

    # Decay exploration
    agent.decay_epsilon()

    # Store rewards
    episode_rewards.append(total_reward)
    episode_wait_times.append(total_wait_time / allocations)
    episode_occupancy_rates.append(env.total_cars_parked / env.total_slots)

    if agent.epsilon <= 0.1 and (best_reward is None or total_reward > best_reward):
        best_reward = total_reward
        best_q_table = copy.deepcopy(agent.q_table)

    print(
        f"Episode {episode + 1} | "
        f"Reward: {total_reward} | "
        f"Epsilon: {agent.epsilon:.3f}"
    )


# Save rewards to CSV
with open("experiments/rewards.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Episode", "Reward"])

    for i, reward in enumerate(episode_rewards):
        writer.writerow([i + 1, reward])


# Plot rewards
plt.plot(episode_rewards)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Rewards Over Episodes")

plt.show()

# Save trained Q-table
with open("models/policy_v1.pkl", "wb") as file:
    pickle.dump(agent.q_table, file)

print("\nPolicy saved successfully!")

# Save best Q-table
if best_q_table is None:
    best_q_table = copy.deepcopy(agent.q_table)

with open("models/policy_best.pkl", "wb") as file:
    pickle.dump(best_q_table, file)

print("Best policy saved successfully!")

# Save experiment log
experiment_log = {
    "avg_reward": round(sum(episode_rewards) / len(episode_rewards), 2),
    "avg_wait_time": round(sum(episode_wait_times) / len(episode_wait_times), 2),
    "occupancy_rate": round(
        sum(episode_occupancy_rates) / len(episode_occupancy_rates),
        2
    ),
    "epsilon": round(agent.epsilon, 2),
    "alpha": round(agent.learning_rate, 2),
    "gamma": round(agent.discount_factor, 2),
    "episodes": episodes,
}

with open("experiments/log.json", "w") as file:
    json.dump(experiment_log, file, indent=4)

print("Experiment log saved successfully!")
