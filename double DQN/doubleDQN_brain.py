from doubleDQN_network import *
from abstract_brain import *
import numpy as np
from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import random

EPSILON = 0.1
EPSILON_MAX = 0.99
LEARNING_RATE = 0.1
REWARD_DECAY = 0.9

class Processing(AbstractBrainPreProcess):
    def __init__(self, frame_size):
        self.deque = deque([np.zeros((frame_size), dtype=np.int) for i in range(4)], maxlen=4)

    def Preprocessing(self, observation):
        frame = rgb2gray(state)
        frame=frame[8:-12,4:-12]            
        frame = frame / 255.0
        frame= transform.resize(frame,[frame_size])
        return frame 

    def four_frame_to_state(self, frame, is_new_episode):
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
        stacked_state = np.stack(self.deque, axis = 2)

        return stacked_state
        
class Learning(AbstractBrainLearning):
    def __init__(self):


    def choose_action(self):

    def memory_replay(self):