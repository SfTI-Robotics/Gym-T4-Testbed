import random
import numpy as np
from collections import deque

import agents.image_input.AbstractBrain as AbstractBrain
from agents.Network import NeuralNetworkBuilder

# =================================================================================================================== #

# TODO: make pretty! Different options for Atari and Basic Control?
# length of transitions deque
MAX_MEMORY_LENGTH = 5000
START_TRAINING = 1000

UPDATE_TARGET_FREQUENCY = 500
MEMORY_REPLAY_FREQUENCY = 1

START_EPSILON_DECAY = 100
MIN_EPSILON = 0.05

# alpha
LEARNING_RATE = 0.001
GAMMA = 0.99

# how many memory's we learn from at a time
BATCH_SIZE = 64

# =================================================================================================================== #


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, is_cartpole: bool):
        super().__init__(observations, actions)
        if is_cartpole:
            self.state_space = (observations, )
            self.network = \
                NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space, LEARNING_RATE)
            self.target_network = \
                NeuralNetworkBuilder.build_cartpole_network(self.state_space, self.action_space, LEARNING_RATE)
        else:
            self.network = \
                NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space, LEARNING_RATE)
            self.target_network = \
                NeuralNetworkBuilder.build_dqn_network(self.state_space, self.action_space, LEARNING_RATE)

        # initialising epsilon changes immediately
        self.e_greedy_formula = 'e = 1-5.45^(-0.007*(episode-1))'
        self.epsilon = 1.0
        self.time_step = 0

        # transitions is where we store memory of max memory length
        self.transitions = deque(maxlen=MAX_MEMORY_LENGTH)
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done, episode):
        self.transitions.append((state, action, reward, next_state, done))
        self.update_epsilon(episode)
        self.time_step = self.time_step + 1

    def update_epsilon(self, episode):
        # equation designed for training on 10 000 episodes
        #  formula = 1 - a ** (-b * (episode - c))
        # self.epsilon = max(MIN_EPSILON, 5.45 ** (-0.009 * (episode - START_EPSILON_DECAY)))
        self.epsilon = max(MIN_EPSILON, 5.45 ** (-0.007 * (episode - START_EPSILON_DECAY)))

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            # this explores by choosing a randomised action
            return random.randrange(self.action_space)
        else:
            # this exploits by choosing your max of your calculated q values
            return np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

    def memory_replay(self):
        # only train network after an initial period, only train at a certain frequency
        if len(self.transitions) < START_TRAINING or self.time_step % MEMORY_REPLAY_FREQUENCY != 0:
            return

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
        target_next = self.target_network.predict(next_states)
        for i in range(BATCH_SIZE):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # bellman equation
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])

        self.network.fit(states, target, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        if self.time_step % UPDATE_TARGET_FREQUENCY == 0:
            self.update_target_model()

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        print('# ====================================== UPDATE ====================================== #')
        self.target_network.set_weights(self.network.get_weights())
