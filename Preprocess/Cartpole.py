from collections import deque
import numpy as np
# http://maciejjaskowski.github.io/2016/03/09/space-invaders.html


class Processing:
    def __init__(self):
        self.deque = deque([np.zeros((110,84), dtype=np.int) for i in range(4)], maxlen=4)
        self.step_max = 400
        self.time_max = 20
        self.reward_min = 0      
        self.reward_max = 400

    def Preprocessing(self, state, is_new_episode):
        return state


    def frames_to_state(self, frame, is_new_episode):

        frame = self.Preprocessing(self, frame)

        return frame


    def get_state_space(self):
        return np.shape(self.deque) 

    def discounted_rewards(self, rewards, gamma):
        return rewards

    def new_action_space(self,actions):
        return actions

    def mapping_actions_to_keys(self,action_taken):            
        return action_taken