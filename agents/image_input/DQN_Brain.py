import random
import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.dqn_networks import build_cartpole_network, build_dqn_network


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)
        # use network suitable for classic control games
        if config['environment'] == 'CartPole-v1':
            self.network = build_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.target_network = \
                build_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
        # use network suitable for Atari games
        else:
            self.network = build_dqn_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.target_network = build_dqn_network(self.state_space, self.action_space, self.config['learning_rate'])

        # initialising epsilon changes immediately
        self.e_greedy_formula = 'e = 1-5.45^(-0.005*(episode-1))'
        self.epsilon = 1.0

        self.update_target_model()

    def update_epsilon(self, episode):
        # equation designed for training on 10 000 episodes
        #  formula = 1 - a ** (-b * (episode - c))
        self.epsilon = \
            max(self.config['min_epsilon'], 5.45 ** (-0.005 * (episode - self.config['initial_epsilon_episodes'])))

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            # this explores by choosing a randomised action
            return random.randrange(self.action_space)
        else:
            # this exploits by choosing your max of your calculated q values
            return np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

    def train_network(self, states, actions, rewards, next_states, dones, episode, step):
        if step % self.config['network_train_frequency'] == 0:
            target = self.network.predict(states)
            target_next = self.target_network.predict(next_states)
            for i in range(self.config['batch_size']):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # bellman equation
                    target[i][actions[i]] = rewards[i] + self.config['gamma'] * np.amax(target_next[i])

            self.network.fit(states, target, batch_size=len(dones), epochs=1, verbose=0)

        if step % self.config['target_update_frequency'] == 0:
            self.update_target_model()
        self.update_epsilon(episode)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())
