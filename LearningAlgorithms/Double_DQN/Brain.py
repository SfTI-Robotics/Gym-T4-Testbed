import numpy as np
import random
from collections import deque

import LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain as AbstractBrain
from LearningAlgorithms.AbstractLearningAlgorithm.Network import NeuralNetworkBuilder

# ========================


class Learning(AbstractBrain.AbstractLearning):

    # length of transitions deque
    MAX_MEMORY_LENGTH = 2000
    START_TRAINING = 1000
    TARGET_MODEL_UPDATE_FREQUENCY = 500
    # how many memory's we learn from at a time
    batch_size = 64

    def __init__(self, observations, actions, is_cartpole):
        super().__init__(observations, actions)

        self.is_cartpole = is_cartpole
        if self.is_cartpole:
            self.initialize_cartpole(observations)
        else:
            self.network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space)
            # use e_greedy formula with longer exploring period

        # copy over weights from behaviour to target
        self.update_target_model()

        self.e_greedy_formula = 'e = min{e_min, e_prev * e_decay}'
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99

        # transitions is where we store memory of max memory length
        self.transitions = deque(maxlen=self.MAX_MEMORY_LENGTH)
        # used to control update frequency of target network
        self.time_step = 0

    def initialize_cartpole(self, observations):
        self.state_space = (observations,)
        self.network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
        # create a new network object for the target network
        self.target_network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space)
        # use e_greedy formula with short exploring period

    def remember(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        # TODO: test this
        # if self.is_cartpole and done:
        #     self.update_target_model()

    def update_epsilon(self, episode):
        # decrease epsilon if learning process has started
        if (self.epsilon > self.epsilon_min) & (self.time_step > self.START_TRAINING):
            self.epsilon = self.epsilon * self.epsilon_decay
            # self.epsilon -= (1 - self.epsilon_min) / 5000000  # boltzman softmax

    # the processed state is used in choosing action
    def choose_action(self, state, episode):
        # TODO: check if this caused deterioration
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_space)
        else:
            action = np.argmax(self.network.predict(np.expand_dims(state, axis=0)))
        self.update_epsilon(episode)
        self.time_step = self.time_step + 1
        return action

    def memory_replay(self):
        # experience replay learning from our memories
        # if len(self.transitions) < self.batch_size:
        if len(self.transitions) < self.START_TRAINING:
            return
        # randomly select 32 memories from 6000
        batch = random.sample(self.transitions, self.batch_size)
        states = np.zeros((self.batch_size,) + self.state_space)
        next_states = np.zeros((self.batch_size,) + self.state_space)
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            states[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = batch[i][3]
            done.append(batch[i][4])
        target = self.network.predict(states)
        target_next = self.network.predict(next_states)
        target_val = self.target_network.predict(next_states)
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # bellman equation
                # target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        self.network.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        # update target network
        if self.time_step % self.TARGET_MODEL_UPDATE_FREQUENCY == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())
