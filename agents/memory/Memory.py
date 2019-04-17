import random
from collections import deque
import numpy as np


# TODO: Which folder?
# TODO: Make this adaptable?
class Memory:
    """
    stores tuples of (state, action, reward, next_state, done) for network-training
    """

    def __init__(self, capacity, state_space):
        self.state_space = state_space
        self.stored_transitions = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        self.stored_transitions.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.stored_transitions, batch_size)
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = batch[i][0]
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states[i] = batch[i][3]
            dones.append(batch[i][4])
        return states, actions, rewards, next_states, dones
