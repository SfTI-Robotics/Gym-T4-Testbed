from abc import ABC, abstractmethod


class AbstractLearning(ABC):

    # variables needed in all Learning classes
    epsilon = 0
    e_greedy_formula = ""

    def __init__(self, observations, actions, config):
        self.state_space = observations
        self.action_space = actions
        self.config = config

        # TODO: not needed for Policy Gradient
        self.e_greedy_formula = 'e = min(e_min, e - e_decay)'
        self.epsilon = self.config['epsilon']
        self.epsilon_decay = (self.config['epsilon'] - self.config['epsilon_min']) / self.config['epsilon_explore']

    @abstractmethod
    def choose_action(self, state, print_predictions=False):
        pass

    @abstractmethod
    def train_network(self, states, actions, rewards, next_states, dones, step):
        pass

    @abstractmethod
    def save_network(self, save_path, model_name, timestamp=None):
        """
        Saves current model to .h5 file, overrides previous model for same environment and algorithm
        :param save_path: path to model folder
        :param model_name: name of model file
        :param timestamp: optional timestamp, if none is specified, current time is used
        """
        pass

    @abstractmethod
    def load_network(self, save_path, model_name):
        """
        Loads previously saved model file to learner.network
        :param save_path: path to model folder
        :param model_name: name of model file
        """
        pass

    def get_predictions(self):
        pass