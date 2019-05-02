import numpy as np

from agents.image_input import AbstractBrain
from agents.networks.actor_critic_networks import build_actor_cartpole_network, build_critic_cartpole_network, \
    build_actor_network, build_critic_network


# TODO: This implementation does not learn for Atari Games!
#   The initial policy heavily favours one action (~99%)
#   After the first network-update, the favourite action has a probability of 1.0, all others have 0
#   Learner proceeds to only choose this one action, regardless of reward
#   implementation according to https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
#   and https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
#       https://github.com/iocfinc/A2C-CartPole/blob/master/A2C%20-%20Cartpole.py (experience replay after every step)

class Learning(AbstractBrain.AbstractLearning):

    value_size = 1

    # create networks
    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        if config['environment'] == 'CartPole-v1':
            self.network = \
                build_actor_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.critic_network = \
                build_critic_cartpole_network(self.state_space, self.value_size, self.config['learning_rate'])
        else:
            self.network = build_actor_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.critic_network = build_critic_network(self.state_space, self.value_size, self.config['learning_rate'])

    def choose_action(self, state):
        # get policy from network
        policy = self.network.predict(np.array([state])).flatten()
        # pick action (stochastic)
        return np.random.choice(self.action_space, 1, p=policy)[0]

    def train_actor(self, states, actions, rewards, next_states, dones):
        advantages = np.zeros((self.config['batch_size'], self.action_space))
        values = self.critic_network.predict(states)
        next_values = self.critic_network.predict(next_states)

        for i in range(self.config['batch_size']):
            if dones[i]:
                advantages[i][actions[i]] = rewards[i] - values[i]
            else:
                # new advantage = r_t + gamma*Q(s', a') - Q(s, a),
                #   see https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
                advantages[i][actions[i]] = rewards[i] + self.config['gamma'] * (next_values[i]) - values[i]

        self.network.fit(states, advantages, epochs=1, verbose=0)

    def train_critic(self, states, rewards, next_states, dones):
        targets = np.zeros((self.config['batch_size'], self.value_size))
        next_values = self.critic_network.predict(next_states)

        for i in range(self.config['batch_size']):
            if dones[i]:
                targets[i][0] = rewards[i]
            else:
                targets[i][0] = rewards[i] + self.config['gamma'] * next_values[i]

        self.critic_network.fit(states, targets, epochs=1, verbose=0)

    def train_network(self, states, actions, rewards, next_states, dones, step):
        # TODO: move this to training-loop to avoid unnecessary sampling
        if step % self.config['network_train_frequency'] == 0:
            # TODO: train functions should be unified to avoid double predictions
            self.train_actor(states, actions, rewards, next_states, dones)
            self.train_critic(states, rewards, next_states, dones)
