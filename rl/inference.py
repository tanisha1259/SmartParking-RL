import pickle
import numpy as np

with open("models/policy_best.pkl", "rb") as f:
    q_table = pickle.load(f)

def allocate_slot(state):
    state = tuple(state)

    if state not in q_table:
        return 0

    q_values = q_table[state]

    return int(np.argmax(q_values))