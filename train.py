from sim.parking_env import ParkingEnvironment
from rl.q_learning import QLearningAgent

import matplotlib.pyplot as plt
import csv
import os
import pickle

# Create folders if not present
os.makedirs("experiments", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Initialize environment
env = ParkingEnvironment(total_slots=10)

# Initialize agent
agent = QLearningAgent(
    num_actions=10
)

# Training settings
episodes = 100

# Store rewards
episode_rewards = []


for episode in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0

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
            next_state
        )

        state = next_state

        total_reward += reward

    # Decay exploration
    agent.decay_epsilon()

    # Store rewards
    episode_rewards.append(total_reward)

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