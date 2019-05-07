import numpy as np
import random

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.dqn_networks import build_dqn_cartpole_network, build_simple_convoluted_net


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        # for classic control environments without any image data
        if self.config['environment'] == 'CartPole-v1':
            self.network = build_dqn_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
            # create a new network object for the target network
            self.target_network \
                = build_dqn_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])

        # for atari games
        else:
            self.network = build_simple_convoluted_net(self.state_space, self.action_space,
                                                       self.config['learning_rate'])
            self.target_network = build_simple_convoluted_net(self.state_space, self.action_space,
                                                              self.config['learning_rate'])

        # copy weights from behaviour to target
        self.update_target_model()

    def update_epsilon(self):
        # self.epsilon \
        #     = max(self.config['min_epsilon'], 5.45 ** (-0.007 * (episode - self.config['initial_epsilon_episodes'])))
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon = max(self.config['epsilon_min'], self.epsilon - self.epsilon_decay)

    # the processed state is used in choosing action
    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            return np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

    def train_network(self, states, actions, rewards, next_states, dones, step):
        if step % self.config['network_train_frequency'] == 0:
            target = self.network.predict(states)
            next_predictions_network = self.network.predict(next_states)
            next_predictions_target = self.target_network.predict(next_states)

            for i in range(self.config['batch_size']):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    a = np.argmax(next_predictions_network[i])
                    target[i][actions[i]] = rewards[i] + self.config['gamma'] * (next_predictions_target[i][a])

            self.network.fit(states, target, batch_size=len(dones), epochs=1, verbose=0)

        # update target network
        if step % self.config['target_update_frequency'] == 0:
            # print('# ========================================== UPDATE ========================================== #')
            self.update_target_model()
        self.update_epsilon()

    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())
