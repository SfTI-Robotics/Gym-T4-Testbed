import numpy as np

class brain_QL:
    def __init__(self, actions, states):
        # self.q_table = np.zeros((states, actions),np.float32)
        self.q_table = np.zeros((states, actions),np.float32)

        self.actions = actions
        self.states = states
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 0.9

    def choose_action(self, env, observation):

        if np.random.rand() > self.epsilon:
            print('goes in')
            # action = np.random.choice(self.q_table[observation,:])
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[observation,:])

     
        return action 

    def learn(self, observation, action, reward, observation_):
        # bellman
        #  Q[s][a] += alpha * ((r + gamma * np.max(Q[ns])) - Q[s][a])
        
        self.q_table[observation, action] += self.alpha*((reward + self.gamma *np.max(self.q_table[observation_,:])) - self.q_table[observation, action])
        print(self.q_table)


     

        
