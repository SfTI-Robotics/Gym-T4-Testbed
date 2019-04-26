from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


class Processor(AbstractProcessor):
    def __init__(self):
        super().__init__()
        # was 400 before, but environment runs for up to 500 steps (and a reward of 498)
        self.step_max = 500
        self.time_max = 5
        self.reward_min = 0      
        self.reward_max = 500

    def preprocessing(self, state, is_new_episode):
        return state

    def frames_to_state(self, frame, is_new_episode):
        return frame

    def get_state_space(self):
        return " No shape"

    def new_action_space(self, actions):
        return actions

    def mapping_actions_to_keys(self, action_taken):
        return action_taken
