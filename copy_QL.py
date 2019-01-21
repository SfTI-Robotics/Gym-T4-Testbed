import numpy as np

class copy_QL:
    def __init__(self, actions, states):
        self.q_table = np.zeros(states, actions)
        self.actions = actions
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 0.9

    def choose_action(self, observation):
        
        if np.random.rand() > self.epsilon:
            action = np.random.choice(self.actions)

        else:
            action = np.argmax(self.actions)


        


