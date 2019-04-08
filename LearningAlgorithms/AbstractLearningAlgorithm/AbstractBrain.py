from abc import ABC, abstractmethod

from LearningAlgorithms.AbstractLearningAlgorithm.Network import NeuralNetworkBuilder


class AbstractLearning(ABC):

    # variables needed in all Learning classes
    epsilon = 0
    gamma = 0
    transitions = None
    e_greedy_formula = ""

    def __init__(self, observations, actions):
        self.state_space = observations
        self.action_space = actions

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def update_epsilon(self, episode):
        pass

    @abstractmethod
    def choose_action(self, state, episode):
        pass

    @abstractmethod
    def memory_replay(self):
        pass
