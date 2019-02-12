from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np
# import cv2
import time


class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(2)], maxlen=2)
        self.step_max = 300
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

        # frame[frame == 144] = 0 # erase background (background type 1)
        # frame[frame == 109] = 0 # erase background (background type 2)
        # frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1

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

      
        # return frame.astype(np.float).ravel()


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
        print('image size: ', np.shape(self.deque))
        return np.shape(self.deque) 
        # return (2,80,80)

        # return 80*80

    def discounted_rewards(self, rewards, gamma):
        # initialise array
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0:
                running_add = 0
            running_add = running_add * gamma + rewards[i]
            discounted_r[i] = running_add

        return discounted_r

    # according to the resource below, actions come in pairs, 
    # therefore, we choose to only pass in one of each pair.
    # this results in only threee actions
    # https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from

    def new_action_space(self,actions):
        # out of 6 actions, only use 3
        return 3 

    def mapping_actions_to_keys(self,action_taken):
        # maps the 3 unique actions to OpenAI's gym action space (6 actions)
        if action_taken == 0:
            action = 0
        elif action_taken == 1:
            action = 2
        elif action_taken == 2:
            action = 3
            
        return action