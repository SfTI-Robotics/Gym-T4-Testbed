import numpy as np
import pandas as pd
import random

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE=0.005

np.random.seed(1)


class QLearning:
    def __init__(self, states, actions):

        self.actions=actions
        self.states=states
        self.q_table = np.zeros((states, actions))
        self.alpha = 0.85
        self.gamma = 0.99
        self.epsilon = 1.0
    
    def choose_action(self, state, episode, env):

        if np.random.rand() > self.epsilon :
            # action = np.random.choice(self.q_table[state,:])
            action = env.action_space.sample()
        else :
            action = np.argmax(self.q_table[state,:])

        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)

        print(action)
        return action

    def learn(self, state, action, reward, state_, done):

        if not done:
            q_target = self.gamma * (reward + np.max(self.q_table[state_,:]) )

        else: 
            q_target = reward

        q_predict = self.q_table[state, action]

        self.q_table[state,action] += self.alpha * (q_target - q_predict )



