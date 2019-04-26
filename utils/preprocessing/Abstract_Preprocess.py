from abc import ABC, abstractmethod
import numpy as np


class AbstractProcessor(ABC):

    def __init__(self):
        # values needed for every processor
        self.step_max = 500
        self.time_max = 120
        self.reward_min = -100
        self.reward_max = 100

    @abstractmethod
    def preprocessing(self, state, is_new_episode) -> any:
        """
        Processes representation of state to reduce its complexity, if possible
        :param state: representation of state
        :param is_new_episode: flag, true if state is first state in an episode, false otherwise
        :return: processed representation of state
        """
        pass

    @abstractmethod
    def frames_to_state(self, frame, is_new_episode) -> any:
        """
        Adds a frame to a state-representation
        :param frame: new frame
        :param is_new_episode: flag, true if state is first state in an episode, false otherwise
        :return: state-representation with added frame
        """
        pass

    @abstractmethod
    def get_state_space(self) -> np.shape:
        """
        Gets shape of state space
        :return: shape of state space
        """
        pass

    @abstractmethod
    def new_action_space(self, actions) -> int:
        """
        Gets size of action space
        :param actions: actions
        :return: size of action space
        """
        pass

    @abstractmethod
    def mapping_actions_to_keys(self, action_taken):
        """
        Maps taken action to an action from the action space
        :param action_taken: taken action
        :return: mapped action
        """
        pass
