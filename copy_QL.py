import numpy as np
import pandas as pd


class copy_QL:
    def __init__(self, actions):
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
        self.actions = actions
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 0.9


    def choose_action(self, observation):
        
        

