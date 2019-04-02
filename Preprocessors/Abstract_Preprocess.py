from abc import ABC, abstractmethod


class AbstractProcessor(ABC):

    # values needed for every processor
    step_max = 0
    time_max = 0
    reward_min = 0
    reward_max = 0

    def __init__(self):
        pass

    @abstractmethod
    def preprocessing(self, state, is_new_episode):
        pass

    @abstractmethod
    def frames_to_state(self, frame, is_new_episode):
        pass

    @abstractmethod
    def get_state_space(self):
        pass

    @abstractmethod
    def discounted_rewards(self, rewards, gamma):
        pass

    @abstractmethod
    def new_action_space(self, actions):
        pass

    @abstractmethod
    def mapping_actions_to_keys(self, action_taken):
        pass
