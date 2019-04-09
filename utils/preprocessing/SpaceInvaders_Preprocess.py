from collections import deque
from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
import numpy as np

from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


# http://maciejjaskowski.github.io/2016/03/09/space-invaders.html
class Processor(AbstractProcessor):
    def __init__(self):
        super().__init__()
        self.deque = deque([np.zeros((110, 84), dtype=np.int) for i in range(4)], maxlen=4)
        self.step_max = 2000
        self.time_max = 30
        self.reward_min = 0
        self.reward_max = 1000

    def preprocessing(self, state, is_new_episode):
        # grayscale
        frame = rgb2gray(state)
        frame = frame[8:-12, 4:-12]
        frame = frame / 255.0
        frame = transform.resize(frame, [110, 84])
        frame = self.frames_to_state(frame, is_new_episode)
        return frame

    def frames_to_state(self, frame, is_new_episode):
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
        else:
            self.deque.append(frame)
        # reshape the deque
        stacked_state = np.stack(self.deque, axis=0)

        return stacked_state

    def get_state_space(self):
        return np.shape(self.deque)

    def discounted_rewards(self, rewards, gamma):
        return rewards

    def new_action_space(self, actions):
        return actions

    def mapping_actions_to_keys(self, action_taken):
        return action_taken
