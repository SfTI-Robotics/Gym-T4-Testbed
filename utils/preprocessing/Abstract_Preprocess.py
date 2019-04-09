from abc import ABC, abstractmethod


class AbstractProcessor(ABC):

    def __init__(self):
        # values needed for every processor
        self.step_max = 500
        self.time_max = 120
        self.reward_min = -100
        self.reward_max = 100

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
