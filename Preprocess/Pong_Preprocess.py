from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np
import cv2
import time


class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(2)], maxlen=1)
        self.step_max = 3000
        self.time_max = 40
        self.reward_min = -35                        
        self.reward_max= 35


    def Preprocessing(self, frame, is_new_episode):
        # grayscale
        # cv2.imshow("image", frame)
       #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        frame = rgb2gray(frame)
        
        frame = frame[35:195]              
        # frame = frame / 255.0
        frame = frame[::2, ::2]
        # frame = transform.resize(frame,[160,160])
    #     cv2.imshow("image", frame)
    #    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        state = self.four_frames_to_state(frame, is_new_episode)
        #img = cv2.imread(state,0)
        
        return state


    def four_frames_to_state(self, frame, is_new_episode):
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            # self.deque.append(frame)
            # self.deque.append(frame)

        else:
            self.deque.append(frame)
        # reshape the deque
        stacked_state = np.stack(self.deque, axis = 0)
        return stacked_state

    def get_state_space(self):
        return np.shape(self.deque) 

    
 
