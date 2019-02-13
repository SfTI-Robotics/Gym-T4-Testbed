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

    def Preprocessing(self, state, is_new_episode):
        # grayscale
        frame = rgb2gray(state)
        frame=frame[8:-12,4:-12]              
        frame = frame / 255.0 
        frame= transform.resize(frame,[110,84])
        frame = self.frames_to_state(frame, is_new_episode)
        return frame


    def frames_to_state(self, frame, is_new_episode):

        # print(frame.shape[:])
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

# rewards can be {-1,0,1}
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