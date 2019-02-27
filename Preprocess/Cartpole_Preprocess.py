from collections import deque
import numpy as np
# http://maciejjaskowski.github.io/2016/03/09/space-invaders.html


class Processing:
    def __init__(self):
        self.step_max = 400
        self.time_max = 5
        self.reward_min = 0      
        self.reward_max = 500

    def Preprocessing(self, state, is_new_episode):
        
        return state


    def frames_to_state(self, frame, is_new_episode):
        
        return frame


    def get_state_space(self):
        return (" No shape") 

    def discounted_rewards(self, rewards, gamma):
        return rewards

    def new_action_space(self,actions):
        return actions

    def mapping_actions_to_keys(self,action_taken):            
        return action_taken