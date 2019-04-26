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

        # TODO: not needed for Policy Gradient
        self.e_greedy_formula = 'e = min(e_min, e - e_decay)'
        self.epsilon = self.config['epsilon']
        self.epsilon_decay = (self.config['epsilon'] - self.config['epsilon_min']) / self.config['epsilon_explore']

    # TODO: not used outside of learner
    # @abstractmethod
    # def update_epsilon(self, episode):
    #     pass

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def train_network(self, states, actions, rewards, next_states, dones, step):
        pass
