import numpy as np
import pandas as pd

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE=0.005



class QLearning:
    def _init_(self, states, actions):

        self.actions=actions
        self.states=states
        self.q_table = np.zeros((actions,states))
        self.alpha = 0.85
        self.gamma = 0.99
        self.epsilon = 1.0
    
    def choose_action(self, state, episode):

        if np.random.rand() < self.epsilon :
            action = np.random.uniform(self.q_table[state,:])
        else :
            action = np.argmax(self.q_table[state,:])

        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)

        return action

    def learn(self, state, action, reward, state_, done):

        if !done:
            q_target = self.gamma * (reward + np.max(self.q_table[state_,:]) )

        else: 
            q_target = reward

        q_predict = self.q_table[state, action]

        self.q_table[,] += self.alpha * (q_target - q_predict )



