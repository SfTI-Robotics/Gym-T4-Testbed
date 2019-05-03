import copy
import random
from collections import deque
import numpy as np


# TODO: Which folder?
# TODO: Make this adaptable! (episodic, prioritized, linear, ...)
class Memory:
    """
    stores tuples of (state, action, reward, next_state, done) for network-training
    """

    def __init__(self, capacity, state_space):
        self.state_space = state_space
        self.capacity = capacity
        self.stored_transitions = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """
        Stores episode in memory
        :param state: state of episode
        :param action: action chosen by agent
        :param reward: reward received for state-action pair
        :param next_state: resulting state of environment
        :param done: flag, true if episode ended after action
        """
        self.stored_transitions.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples batch_size random episodes from memory
        :param batch_size: amount of random samples
        :return: states, actions, rewards, next_states, dones of randomly sampled episodes
        """
        # update batch size in case memory doesn't contain enough values
        batch_size = min(len(self.stored_transitions), batch_size)

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

    def sample_all(self):
        """
        Gets all episodes from memory in order of occurrence, clears memory
        :return: states, actions, rewards, next_states, dones of all episodes in order of occurrence
        """
        batch = copy.deepcopy(self.stored_transitions)
        batch_size = len(batch)
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = batch[i][0]
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states[i] = batch[i][3]
            dones.append(batch[i][4])

        self.stored_transitions = deque(maxlen=self.capacity)

        return states, actions, rewards, next_states, dones

    def sample_last(self):
        """
        Gets most recently added episode from memory
        :return: state, action, reward, next_state and done of most recent episode
        """
        batch_size = 1
        pos = len(self.stored_transitions) - 1
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = self.stored_transitions[pos][0]
            actions.append(self.stored_transitions[pos][1])
            rewards.append(self.stored_transitions[pos][2])
            next_states[i] = self.stored_transitions[pos][3]
            dones.append(self.stored_transitions[pos][4])

        return states, actions, rewards, next_states, dones
