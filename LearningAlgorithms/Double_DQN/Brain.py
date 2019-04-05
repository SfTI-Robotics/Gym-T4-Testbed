import numpy as np
import random
from collections import deque

import LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain as AbstractBrain
from LearningAlgorithms.AbstractLearningAlgorithm.Network import NeuralNetworkBuilder

# ========================

# length of transitions deque
MAX_MEMORY_LENGTH = 1000
START_TRAINING = 500
# how many memory's we learn from at a time
batch_size = 64


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, is_cartpole):
        super().__init__(observations, actions)

        if is_cartpole:
            self.state_space = (observations,)
            self.network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
        else:
            self.network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)

        # copy over weights from behaviour to target
        self.update_target_model()

        self.epsilon = 0
        self.gamma = 0.95
        # self.e_greedy_formula = 'e = 1-1.2^(-0.003*(episode-2500))'
        self.e_greedy_formula = 'e = 1-5.45^(-0.009*(episode-100))'

        # transitions is where we store memory of max memory length
        self.transitions = deque(maxlen=MAX_MEMORY_LENGTH)

    def remember(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))

    def update_epsilon(self, episode):
        # increase epsilon
        #  formula = 1 - a ** (-b * (episode - c))
        # self.epsilon = 1 - 1.2 ** (-0.003 * (episode - 2500))
        self.epsilon = 1 - 5.45 ** (-0.009 * (episode - 100))

    # the processed state is used in choosing action
    def choose_action(self, state, episode):
        if random.random() > self.epsilon:
            action = random.randrange(self.action_space)
        else:
            action = np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

        self.update_epsilon(episode)

        return action

    def memory_replay(self):
        # experience replay learning from our memories
        if len(self.transitions) < MAX_MEMORY_LENGTH:
            return
        # randomly select 32 memories from 6000
        batch = random.sample(self.transitions, batch_size)
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        action, reward, done = [], [], []

        for i in range(batch_size):
            states[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = batch[i][3]
            done.append(batch[i][4])
        target = self.network.predict(states)
        target_next = self.target_network.predict(next_states)
        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # bellman equation
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])
        self.network.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

        ###############################################################################################################
        '''
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # resize array by increasing dimension
                next_state = np.expand_dims(next_state, axis=0)
                # bootstrapping the predicted reward as Q-value
                target = reward + self.gamma * np.max(self.target_network.model.predict(next_state))

            # resize array by increasing dimension
            state = np.expand_dims(state, axis=0)
            target_f = self.network.model.predict(state)

            target_f[0][action] = target
            # print('target_f =', target_f)
            self.network.model.fit(state, target_f, verbose=0)
        '''
        ###############################################################################################################
        # self.update_target_model()

    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())

    def finish_episode_training(self):
        self.update_target_model()


# # experience replay
# batch = random.sample(self.transitions, batch_size)
# # initialise arrays
# states = np.zeros((batch_size, *self.state_space))
# next_states = np.zeros((batch_size, *self.state_space))
# action, reward, done = [], [], []

# # extract variables from transition
# # extract separate s,a,r.s'
# for i in range(batch_size):
#     states[i] = np.array(batch[i][0])
#     action.append(batch[i][1])
#     reward.append(batch[i][2])
#     next_states[i] = np.array(batch[i][3])
#     done.append(batch[i][4])

# target = self.network.model.predict(states, batch_size=batch_size)
# target_next = self.network.model.predict(next_states, batch_size=batch_size)
# target_value = self.target_network.model.predict(next_states, batch_size=batch_size)

# for sample in range(batch_size):
#     # check if transition was at end of episode
#     is_done = done[sample]
#     if is_done:
#         target[sample][action[sample]] = reward[sample]
#     else:
#         # Bellman Equation
#         target[sample][action[sample]] = reward[sample] + self.gamma * np.max(target_value[sample])

# # calculates loss and does optimisation
# # run graph self.target_model.set_weights(self.model.get_weights())
# self.network.model.fit(states, target, batch_size=batch_size,
# epochs=1, verbose=0)
