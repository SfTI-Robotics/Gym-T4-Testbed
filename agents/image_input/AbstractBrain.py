from abc import ABC, abstractmethod


class AbstractLearning(ABC):

    # variables needed in all Learning classes
    epsilon = 0
    e_greedy_formula = ""
    network = None

    def __init__(self, observations, actions, config):
        self.state_space = observations
        self.action_space = actions
        self.config = config

    @abstractmethod
    def update_epsilon(self, episode):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def train_network(self, states, actions, rewards, next_states, dones, episode, step):
        pass
