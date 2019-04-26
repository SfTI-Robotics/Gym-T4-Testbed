from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


class Processor(AbstractProcessor):

    def __init__(self):
        super().__init__()
        self.deque = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
        self.step_max = 2000
        self.time_max = 30
        self.reward_min = 0
        self.reward_max = 1000

    def preprocessing(self, frame, is_new_episode) -> np.ndarray:
        # see https://github.com/gsurma/atari/blob/master/gym_wrappers.py
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84])
        return self.frames_to_state(x_t.astype(np.uint8), is_new_episode)

    def frames_to_state(self, frame, is_new_episode):
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
        else:
            self.deque.append(frame)

            plt.imshow(frame, cmap="gray")
            plt.show()
        # reshape the deque
        stacked_state = np.stack(self.deque, axis=0)

        return stacked_state

    def get_state_space(self):
        return np.shape(self.deque)

    def new_action_space(self, actions):
        return actions

    def mapping_actions_to_keys(self, action_taken):
        return action_taken
