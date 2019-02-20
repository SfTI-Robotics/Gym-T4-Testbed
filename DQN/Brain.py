import numpy as np
import random
from collections import deque
# import the network class from file
from DQN.Network import neural_net
import time

# ========================

# length of transitions deque
MAX_MEMORY_LENGTH = 60
# alpha
LEARNING_RATE = 0.1
# gamma
REWARD_DECAY = 0.8
# how many memory's we learn from at a time
batch_size = 32

class Learning():


    def __init__(self,observations, actions):
        self.state_space = observations
        self.action_space = actions
        #initialise network here and not in the main file
        #length of action space should be whatever new_action_space outputs see preprocess for more
        self.network = neural_net(self.state_space, self.action_space)
        #initialiseing epsilon changes immmediatley
        self.epsilon = 0
        self.e_greedy_formula = 'e-greedy formula = '
        self.gamma = REWARD_DECAY
        self.alpha = 0.86
        #transitions is where we store memory of max memory length
        self.transitions = deque(maxlen = MAX_MEMORY_LENGTH)


    def choose_action(self, state, episode):
        if random.random() > self.epsilon:
            #this expolres by choosing a randomised action
            action = random.randrange(self.action_space)
            # print('EXPLORE')
        else:
            #this exploits by choosing your max of your calculated q values
            action = np.argmax(self.network.model.predict(np.expand_dims(state, axis = 0)))
            # print('EXPLOIT')

        # increase epsilon
        # equation designed for training on 10 000 episodes
        # epsilon is below 0 until 'c' episodes is reached and is approx 1 for last 1000 episodes
        #  formula = 1 - a ** (-b * (episode - c))
        self.epsilon = 1 - 1.2 ** (-0.003 * (episode - 2500))
        self.e_greedy_formula = 'e = 1-1.2^(-0.003*(episode-2500))'

        return action


    def memory_replay(self):
        #this is highly inefficient
        # experience replay learning from our memories once there are 5000 memories
        # condition for how many transitions need to stored before meory replay is used(1 step=1 transtion)
        if len(self.transitions) < MAX_MEMORY_LENGTH:
            return
        # randomly select 32 memories from 5000
        batch = random.sample(self.transitions, batch_size)

        # for state, action, reward, next_state, done in batch:
        #     target = reward
        #     if not done:
        #         # resize array by increasing dimension
        #         next_state = np.expand_dims(next_state, axis= 0)
        #         # bootstrapping the predicted reward as Q-value
        #         target = reward + self.gamma * np.max(self.network.model.predict(next_state))

        #     # resize array by increasing dimension
        #     state = np.expand_dims(state, axis=0)
        #     target_f = self.network.model.predict(state)

        #     target_f[0][action] = target
        #     # print('target_f =', target_f)
        #     self.network.model.fit(state, target_f, verbose = 0)
       
    #     =============================================
        state_array = np.zeros((batch_size, *self.state_space)) 
        next_state_array = np.zeros((batch_size, *self.state_space))
        out = np.zeros((batch_size,3))
        batch = random.sample(self.transitions, batch_size)
        i = 0
        for state, action, reward, next_state, done in batch:
            state_array[i:i+1] = state
            next_state_array[i:i+1] = next_state
            target = reward

            if done == False:
                next_state = np.expand_dims(next_state, axis=0)
                target = reward + self.gamma * np.amax(self.network.model.predict(next_state)[0] )
            
            state = np.expand_dims(state, axis=0)
            out[i] = self.network.model.predict(state)
            out[i][action] = target
            i += 1

        self.network.model.fit(state_array,out,batch_size = batch_size, epochs=1,verbose=1)
       
    #     =============================================



        print("finish replay")
