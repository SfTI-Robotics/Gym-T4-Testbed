import random
import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.actor_critic_networks import build_actor_cartpole_network, build_actor_network
from agents.networks.dqn_networks import build_dqn_cartpole_network, build_dqn_network


class SpiralQLearner:
    def __init__(self, observations, actions, config):
        self.state_space = observations
        self.action_space = actions
        self.config = config
        self.e_greedy_formula = 'e = min(e_min, e - e_decay)'
        self.epsilon = self.config['epsilon']
        self.epsilon_decay = (self.config['epsilon'] - self.config['epsilon_min']) / self.config['epsilon_explore']
        # use network suitable for classic control games
        if config['environment'] == 'CartPole-v1':
            self.network = build_dqn_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.target_network = \
                build_dqn_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
        # use network suitable for Atari games
        else:
            self.network = build_dqn_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.target_network = build_dqn_network(self.state_space, self.action_space, self.config['learning_rate'])
        self.update_target_model()

    def update_epsilon(self):
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon = max(self.config['epsilon_min'], self.epsilon - self.epsilon_decay)

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            # this explores by choosing a randomised action
            return random.randrange(self.action_space)
        else:
            # this exploits by choosing your max of your calculated q values
            return np.argmax(self.network.predict(np.expand_dims(state, axis=0)))

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_network.set_weights(self.network.get_weights())

    def train_network(self, states, actions, rewards, next_states, dones, step):
        if step % self.config['network_train_frequency'] == 0:
            # TODO: train network with data collected by policy gradient and/or data collected by q-learner?
            #   potentially, normal DQN_Brain can be used here, just adapt memory and coordinate in hybrid brain
            pass
        if step % self.config['target_update_frequency'] == 0:
            self.update_target_model()
        self.update_epsilon()


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        self.dqn_agent = SpiralQLearner(observations, actions, config)
        self.pg_network = self.build_pg_network()

        self.switch = False
        self.train_counter = 0
        self.switch_point = 100

    def choose_action(self, state):
        if self.switch:
            return self.dqn_agent.choose_action(state)
        else:
            policy = self.network.predict(np.array([state])).flatten()
            return np.random.choice(np.arange(self.action_space), 1, p=policy)[0]

    def train_network(self, states, actions, rewards, next_states, dones, step):
        if self.switch:
            # TODO: train q-learner or hybrid actor-critic version
            pass
        else:
            # TODO: train policy gradient or hybrid actor-critic version
            pass

    def evaluate_policy(self):
        if self.switch:
            # TODO: evaluate q-learner,
            #   switch to policy-gradient or to hybrid actor-critic version if plateau was reached
            pass
        else:
            # TODO: evaluate policy gradient,
            #   switch to q-learner if plateau was reached
            pass

    def build_pg_network(self):
        if self.config['environment'] == 'CartPole-v1':
            return build_actor_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
        else:
            return build_actor_network(self.state_space, self.action_space, self.config['learning_rate'])
