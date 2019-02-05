import numpy as np
import random
from collections import deque


MAX_MEMORY_LENGTH = 5000
LEARNING_RATE = 0.01
REWARD_DECAY = 0.9
START_TRAINING = 2000
batch_size=64

class Learning():


    def __init__(self,observations, actions, network):
        self.state_space = observations 
        self.action_space = actions

        self.network = network
        # self.net = neural_net(self.observation_space, self.action_space)
        
        self.epsilon = 1.0
        self.gamma = 0.95
        self.alpha = 0.86
        #transitions is where we store memory of max memory length
        self.transitions = deque(maxlen = MAX_MEMORY_LENGTH)

    # the processed state is used in choosing action
    def choose_action(self, state, episode):
        if random.random() > self.epsilon:
            action = random.randrange(self.action_space)
        else:
            action = np.argmax(self.network.model.predict(np.expand_dims(state, axis = 0)))

        # decay epsilon
        self.epsilon = 1 - 10 ** (-0.003 * episode)

        return action



    def memory_replay(self):
        # experience replay learning from our memories
        if len(self.transitions) < START_TRAINING:
            return

        # experience replay
        batch = random.sample(self.transitions, batch_size)
###############################################################################################
        # initialise arrays
        states = np.zeros((batch_size, *self.state_space)) 
        next_states = np.zeros((batch_size, *self.state_space))
        action, reward, done = [], [], []
        


        # extract variables from transition
        # extract seperate s,a,r.s'
        for i in range(batch_size):
            states[i] = np.array(batch[i][0])
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = np.array(batch[i][3])
            done.append(batch[i][4])  

        target = self.network.model.predict(states, batch_size=batch_size)
        target_next = self.network.model.predict(next_states, batch_size=batch_size)
###############################################################################################

        for sample in range(batch_size):
            # check if transition was at end of episode
            is_done = done[sample]
            if is_done:
                target[sample][action[sample]] = reward[sample]
            else:
                # Bellman Equation
                target[sample][action[sample]] = reward[sample] + self.gamma * np.max(target_next[sample])

        # calculates loss and does optimisation
        # run graph
        self.network.model.fit(states, target, batch_size=batch_size,
        epochs=1, verbose=0)