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

TARGET_UPDATE_FREQUENCY = 3000


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, is_cartpole):
        super().__init__(observations, actions)

        self.is_cartpole = is_cartpole
        if self.is_cartpole:
            self.state_space = (observations,)
            self.network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
            # use e_greedy formula with short exploring period
            self.e_greedy_formula = 'e = 1-5.45^(-0.009*(episode-100))'
        else:
            self.network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)
            # use e_greedy formula with longer exploring period
            self.e_greedy_formula = 'e = 1-1.2^(-0.003*(episode-2500))'

        # copy over weights from behaviour to target
        self.update_target_model()

        self.epsilon = 0
        self.gamma = 0.95

        # transitions is where we store memory of max memory length
        self.transitions = deque(maxlen=MAX_MEMORY_LENGTH)

        # used to control update frequency of target network
        self.time_step = 0

    def remember(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))

    def update_epsilon(self, episode):
        # increase epsilon
        #  formula = 1 - a ** (-b * (episode - c))
        if self.is_cartpole:
            self.epsilon = 1 - 5.45 ** (-0.009 * (episode - 100))
        else:
            self.epsilon = 1 - 1.2 ** (-0.003 * (episode - 2500))

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
        # if len(self.transitions) < batch_size:
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
        target_next = self.network.predict(next_states)
        target_val = self.target_network.predict(next_states)
        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # bellman equation
                # target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        self.network.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)
        # update target network
        if self.time_step % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_model()
        self.time_step = self.time_step + 1

    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())

    def finish_episode_training(self):
        self.update_target_model()

