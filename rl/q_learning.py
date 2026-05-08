import random
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):

        self.num_actions = num_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = {}

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        return self.q_table[state]

    def choose_action(self, state, available_actions):

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)

        q_values = self.get_q_values(state)

        filtered_q_values = {
            action: q_values[action]
            for action in available_actions
        }
        return max(filtered_q_values, key=filtered_q_values.get)

    def update_q_table(self, state, action, reward, next_state):

        current_q = self.get_q_values(state)[action]

        max_future_q = np.max(self.get_q_values(next_state))

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        self.q_table[state][action] = new_q

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay