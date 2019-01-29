import numpy as np
import gym
from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import random
class Processing():
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)

    def Preproccesing(self, state):
        # grayscale
        frame = rgb2gray(state)   
        frame=frame[35:195]                    
        frame = frame / 255.0
        frame= transform.resize(frame,[80,80])                    



    def four_frame_to_state(self, state, is_new_episode):
        
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
        stacked_state = np.stack(self.deque, axis = 2)

        return stacked_state                        

class Learning():

    def __init__(self, ):
        
        self.epsilon = 1.0


    def choose_action(self, state, episode):
        if random.random() < self.epsilon: 
            action = 0
        else: 
            action = 0

        # decay epsilon
        self.epsilon = 0.1 + (0.99+0.1) * np.exp(-0.995 * episode)

        return action



    def memory_replay(self, ):

                        







                    
                                
        

        

