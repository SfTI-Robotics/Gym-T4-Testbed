import numpy as np
import random
import keras
import keras.models
from collections import deque
from Double_DQN.Network import neural_net


MAX_MEMORY_LENGTH = 5000
LEARNING_RATE = 0.01
REWARD_DECAY = 0.9
START_TRAINING = 500
batch_size=32

class Learning():


    def __init__(self,observations, actions):
        self.state_space = observations 
        self.action_space = actions

        # behaviour network instantiated in the run_main.py file
        self.network = neural_net(self.state_space, self.action_space)
        # create a new network object for the target network
        self.target_network = neural_net(self.state_space, self.action_space)
        # copy over weights from behaviour to target
        self.update_target_model()
        
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
        target_value = self.target_network.model.predict(next_states, batch_size=batch_size)

        for sample in range(batch_size):
            # check if transition was at end of episode
            is_done = done[sample]
            if is_done:
                target[sample][action[sample]] = reward[sample]
            else:
                # Bellman Equation
                target[sample][action[sample]] = reward[sample] + self.gamma * np.max(target_value[sample])

        # calculates loss and does optimisation
        # run graphself.target_model.set_weights(self.model.get_weights())
        self.network.model.fit(states, target, batch_size=batch_size,
        epochs=1, verbose=0)

        self.update_target_model()
        
    def update_target_model(self):
            self.target_network.model.set_weights(self.network.model.get_weights())


