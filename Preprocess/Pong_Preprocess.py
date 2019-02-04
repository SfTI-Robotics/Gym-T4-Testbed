from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np


class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
        self.step_max = 3000
        self.time_max = 40
        self.reward_min = -35                        
        self.reward_max= 35


    def Preprocessing(self, state):
        # grayscale
        frame = rgb2gray(state)
        frame=frame[35:195]              
        frame = frame / 255.0
        frame= transform.resize(frame,[80,80])
        return frame


    def four_frames_to_state(self, state, is_new_episode):
        frame = self.Preprocessing(state)
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)

        else:
            self.deque.append(frame)
        # reshape the deque
        stacked_state = np.stack(self.deque, axis = 0)
        return stacked_state

    def get_state_space(self):
        return np.shape(self.deque) 

    
 