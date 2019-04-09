import numpy as np
import random
from collections import deque

import LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain as AbstractBrain
from LearningAlgorithms.AbstractLearningAlgorithm.Network import NeuralNetworkBuilder

# =================================================================================================================== #

# length of transitions deque
MAX_MEMORY_LENGTH = 2000
START_TRAINING = 1000

TARGET_MODEL_UPDATE_FREQUENCY = 500

START_EPSILON_DECAY = 100

LEARNING_RATE = 0.001

GAMMA = 0.99

# how many memory's we learn from at a time
# TODO: reduce batch size?
BATCH_SIZE = 64

# =================================================================================================================== #


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, is_cartpole):
        super().__init__(observations, actions)

        # for classic control environments without any image data
        self.is_cartpole = is_cartpole
        if self.is_cartpole:
            self.state_space = (observations,)
            self.network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space,
                                                                       LEARNING_RATE)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space,
                                                                              LEARNING_RATE)

        # for atari games
        else:
            self.network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space,
                                                                  LEARNING_RATE)
            # create a new network object for the target network
            self.target_network = NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space,
                                                                         LEARNING_RATE)
            # use e_greedy formula with longer exploring period

        # copy weights from behaviour to target
        self.update_target_model()

        self.e_greedy_formula = 'e = 1-5.45^(-0.009*(episode-100))'
        self.epsilon = 1.0

        # transitions is where we store memory of max memory length
        self.transitions = deque(maxlen=MAX_MEMORY_LENGTH)

        # used to control update frequency of target network
        self.time_step = 0

    def remember(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))

    def update_epsilon(self, episode):
        # equation designed for training on 10 000 episodes
        # epsilon is below 0 until 'c' episodes is reached and is approx 1 for last 1000 episodes
        #  formula = 1 - a ** (-b * (episode - c))
        self.epsilon = 5.45 ** (-0.009 * (episode - START_EPSILON_DECAY))

    # the processed state is used in choosing action
    def choose_action(self, state, episode):
        # if random.random() <= self.epsilon:
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_space)
        else:
            action = np.argmax(self.network.predict(np.expand_dims(state, axis=0)))
        self.update_epsilon(episode)
        self.time_step = self.time_step + 1
        return action

    def memory_replay(self):
        # TODO: when should we train? after every action? after ever episode? after n steps?
        # TODO: when should we start training? once we have one full batch? once the memory is full? after n steps?
        # experience replay learning from our memories
        # if len(self.transitions) < self.batch_size:
        if len(self.transitions) < START_TRAINING:
            return
        # randomly select 32 memories from 6000
        batch = random.sample(self.transitions, BATCH_SIZE)
        states = np.zeros((BATCH_SIZE,) + self.state_space)
        next_states = np.zeros((BATCH_SIZE,) + self.state_space)
        action, reward, done = [], [], []

        for i in range(BATCH_SIZE):
            states[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.network.predict(states)
        target_next = self.network.predict(next_states)
        target_val = self.target_network.predict(next_states)

        for i in range(BATCH_SIZE):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + GAMMA * (target_val[i][a])

        self.network.fit(states, target, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        # update target network
        if self.time_step % TARGET_MODEL_UPDATE_FREQUENCY == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())
