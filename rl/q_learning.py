import random
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05
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
        best_q_value = max(filtered_q_values.values())
        best_actions = [
            action
            for action, q_value in filtered_q_values.items()
            if q_value == best_q_value
        ]

        return random.choice(best_actions)

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        next_available_actions=None,
        done=False
    ):

        current_q = self.get_q_values(state)[action]

        if done:
            max_future_q = 0
        else:
            next_q_values = self.get_q_values(next_state)

            if next_available_actions:
                max_future_q = max(
                    next_q_values[next_action]
                    for next_action in next_available_actions
                )
            else:
                max_future_q = np.max(next_q_values)

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        self.q_table[state][action] = new_q

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon * self.epsilon_decay,
                self.epsilon_min
            )
