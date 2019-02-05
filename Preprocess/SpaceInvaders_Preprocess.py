from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np
# http://maciejjaskowski.github.io/2016/03/09/space-invaders.html


class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((110,84), dtype=np.int) for i in range(4)], maxlen=4)
        self.step_max = 3000
        self.time_max = 100
        self.reward_min = -10         
        self.reward_max= 1000 

    def Preprocessing(self, state):
        # grayscale
        frame = rgb2gray(state)
        frame=frame[8:-12,4:-12]              
        frame = frame / 255.0 
        frame= transform.resize(frame,[110,84])
        return frame


    def four_frames_to_state(self, state, is_new_episode):
        
        frame = self.Preprocessing(state)
        print(frame.shape[:])
        # frame = self.Preprocessing(self, state)
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
