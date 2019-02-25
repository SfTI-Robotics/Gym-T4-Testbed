from collections import deque
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np
#uncomment if you want to see what you've process and if it getting the right input
# import cv2
import matplotlib.pyplot as plt
import time


class Processing:
    def __init__(self):
        # deque is used to temporaily hold images while it being stacked        
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(2)], maxlen=2)
        # This stuff is need for the graph this needs to be set up for each environment
        # We decided to put it in preprocess as for each environment:
        #             the range of steps a game takes changes 
        #         but the lowest is always zero so only max is entered here
        #   same for time min will always be zero while time max varies from env to env
        #              reward min and max both vary        
        self.step_max = 2200
        self.time_max = 40
        self.reward_min = -30                        
        self.reward_max= 30

    def Preprocessing(self, frame, is_new_episode):
        # this is to see if your processedframed looks alright
        # the part below show the original frame        
        # cv2.imshow("image", frame)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()

        # grayscale
        frame = rgb2gray(frame)
        
        #cropping frame
        # this is specific to pong and the main reason we decided to 
        # set up a proccesing file instaed of doing it in the main
        frame = frame[35:195]        


        # frame = frame / 255.0
        # extract every second pixel in the image and resizes it to 80 x 80 
        frame = frame[::2, ::2]
        
        
        # uncomment to see single processed frame
        plt.imshow(np.array(frame))
        plt.show()
        # cv2.imshow("image", np.array(frame))
        # cv2.waitKey(10)
    #     cv2.destroyAllWindows()
        
        # state is a deque of 2 frames
        state = self.frames_to_state(frame, is_new_episode)
        
        return state

    def frames_to_state(self, frame, is_new_episode):
        
        if is_new_episode:
            # all frames in new deque are of same state
            # new epsisode the deque is completely empty so both slots need to be filled
            self.deque.append(frame)
            self.deque.append(frame)

        else:
            # append new frame to deque
            # in other steps one slot is already filled and the second one needs to be filled
            self.deque.append(frame)
            
        # reshape the deque so that network recognises that its two frames stacked toghether
        stacked_state = np.stack(self.deque, axis = 0)
        return stacked_state

    def get_state_space(self):
        # this is used to return the processed shape  to the network initialiser
        # print('image size: ', np.shape(self.deque))
        return np.shape(self.deque) 


# In Pong rewards can be {-1,0,1}
# when -1 or 1 game has been reset
# the reward is backpropagated so that the actions leading up to the result 
# are assigned a percentage of the reward.
# 'running add' variable that resets to 0 whenever this happens


    def discounted_rewards(self, rewards, gamma):
        # initialise array and variable
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        # np.zeroes_like(rewards) creates an array the same shape as the input
        running_add = 0

        # starting from the last element, the reward of each action is changed to be a percentage of 1 or -1
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0:
                running_add = 0
            running_add = running_add * gamma + rewards[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r) #normalizing the result
        discounted_r /= np.std(discounted_r) 
        return discounted_r

    # according to the resource below, Pong-v0 actions come in pairs, 
    # therefore, we choose to only pass in one of each pair.
    # this results in only threee actions
    # https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from

    def new_action_space(self,actions):
        # out of 6 actions, only use 3
        # 0 and 1 is taking no move
        # 2 and 4 move up 
        # 3 and 5 move down
        # we want it to only be able to select 0,2,3
        return 3

    def mapping_actions_to_keys(self,action_taken):
        # maps the 3 unique actions to OpenAI's gym action space (6 actions)
        # the actions it takes from main are 0-2
        # 0=0(no move)
        # 1=2(up)
        # 2=3(down)

        if action_taken == 0:
            action = 0
        elif action_taken == 1:
            action = 2
        elif action_taken == 2:
            action = 3
            
        return action