import numpy as np
import random

class Q_Learning:
    def __init__(
            self, 
            state_num, 
            action_num
            ):
        self.state_num = state_num
        self.action_num = action_num

        self.q_table = np.zeros((state_num,action_num))
        self.reward_decay = 0.99
        self.learning_rate = 0.5
        self.epsilon = 0.1
        self.epsilon_max = 0.99
        self.epsilon_decay = 0.998

    def learn(self, obs, action, reward, obs_, done):
        if done:
            q_value = reward
        else:
            q_value = reward + self.reward_decay * np.max(self.q_table[obs_,:])

        self.q_table[obs,action] = self.learning_rate * (self.q_table[obs,action] - q_value)

        # print(self.q_table)

    def choose_action(self, obs, _, env):
        if random.uniform(0, 1) > self.epsilon:
            action = env.action_space.sample()
            # print(action)
        else:
            action = np.argmax(self.q_table[obs, :])
            # print(action)
        
        if self.epsilon >= self.epsilon_max:
            self.epsilon = self.epsilon_max
        else:
            self.epsilon /= self.epsilon_decay
            # print(self.epsilon)

        return action



