from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np



class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)

    def Preprocessing(self, state):
        # grayscale
        frame = rgb2gray(state)
        frame=frame[35:195]              
        frame = frame / 255.0
        frame= transform.resize(frame,[80,80])
        # print('frame=', frame.shape[:])
        return frame


    def four_frames_to_state(self, state, is_new_episode):
        # print('state  ', np.shape(state))
        frame = self.Preprocessing(state)
        # print('frame = ', frame.shape[:])
        # frame = self.Preprocessing(self, state)
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)

        else:
            self.deque.append(frame)
        # print('deque=', np.shape(self.deque))
        # reshape the deque
        stacked_state = np.stack(self.deque, axis = 0)
        # print('stack=', np.shape(stacked_state))
        return stacked_state

    def get_state_space(self):
        return np.shape(self.deque) 
        # .shape[:]
