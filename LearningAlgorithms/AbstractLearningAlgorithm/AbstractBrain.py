from abc import ABC, abstractmethod

from LearningAlgorithms.AbstractLearningAlgorithm.Network import NeuralNet


class AbstractLearning(ABC):

    # variables needed in all Learning classes
    epsilon = 0
    gamma = 0
    transitions = None
    e_greedy_formula = ""

    def __init__(self, observations, actions):
        self.state_space = observations
        self.action_space = actions

        # initialise network here and not in the main file
        # length of action space should be whatever new_action_space outputs see preprocess for more
        self.network = NeuralNet(self.state_space, self.action_space)

    @abstractmethod
    def choose_action(self, state, episode):
        pass

    @abstractmethod
    def memory_replay(self, episode):
        pass
