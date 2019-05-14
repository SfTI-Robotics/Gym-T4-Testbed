import datetime
import os
import sys

import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.ppo_networks import build_ppo_critic_network, build_ppo_actor_network


# see https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        if self.config['environment'] == 'CartPole-v1':
            self.actor_network \
                = build_ppo_actor_network(self.state_space, self.action_space, self.config['learning_rate'],
                                          self.config['clipping_loss_ratio'], self.config['entropy_loss_ratio'])
            self.critic_network = build_ppo_critic_network(self.state_space, self.config['learning_rate'])
        else:
            self.actor_network \
                = build_ppo_actor_network(self.state_space, self.action_space, self.config['learning_rate'],
                                          self.config['clipping_loss_ratio'], self.config['entropy_loss_ratio'])
            self.critic_network = build_ppo_critic_network(self.state_space, self.config['learning_rate'])

        self.dummy_action, self.dummy_value = np.zeros((1, self.action_space)), np.zeros((1, 1))

        self.old_predictions = []

    def choose_action(self, state):
        p = self.actor_network.predict([np.array([state]), self.dummy_value, self.dummy_action])
        self.old_predictions.append(p)
        return np.random.choice(self.action_space, p=np.nan_to_num(p[0]))

    def compute_advantage(self, states, rewards, next_states, dones):
        advantages = np.zeros_like(rewards)
        previous_advantage = 0
        next_values = self.critic_network.predict(next_states)
        current_values = self.critic_network.predict(states)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                previous_advantage = 0
                current_advantage = rewards[t] + - current_values[t]
            else:
                current_advantage = rewards[t] + self.config['gamma'] * next_values[t] - current_values[t]
            advantages[t] = current_advantage + self.config['gamma'] * previous_advantage
            previous_advantage = advantages[t]
        return advantages

    def compute_critic_targets(self, rewards, next_states, dones):
        targets = np.zeros_like(rewards)
        next_values = self.critic_network.predict(next_states)
        for i in range(len(rewards)):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.config['gamma'] * next_values[i]
        return targets

    def train_network(self, states, actions, rewards, next_states, dones, step):
        advantages = self.compute_advantage(states, rewards, next_states, dones)
        # advantage = (advantage - advantage.mean()) / advantage.std()
        targets = self.compute_critic_targets(rewards, next_states, dones)

        old_predictions = np.zeros((len(dones), self.action_space))
        actions_taken = np.zeros((len(dones), self.action_space))
        for i in range(len(dones)):
            old_predictions[i] = self.old_predictions[i]
            actions_taken[i][actions[i]] = 1

        self.actor_network.fit([states, advantages, old_predictions], [actions_taken],
                               batch_size=self.config['batch_size'], shuffle=True, epochs=self.config['epochs'],
                               verbose=False)
        self.critic_network.fit([states], [targets], batch_size=self.config['batch_size'], shuffle=True,
                                epochs=self.config['epochs'], verbose=False)
        self.old_predictions = []

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path + 'actors/'):
            os.makedirs(save_path + 'actors/')
        if not os.path.exists(save_path + 'critics/'):
            os.makedirs(save_path + 'critics/')
        # set timestamp if none was specified
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.actor_network.save_weights(save_path + 'actors/' + model_name + '_' + timestamp + '.h5', overwrite=True)
        self.critic_network.save_weights(save_path + 'critics/' + model_name + '_' + timestamp + '.h5', overwrite=True)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path + 'actors/') and os.path.exists(save_path + 'critics/'):
            self.actor_network.load_weights(save_path + 'actors/' + model_name)
            self.critic_network.load_weights(save_path + 'critics/' + model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")

    def get_test_learner(self):
        test_learner = Learning(self.state_space, self.action_space, self.config)
        # use current network weights for testing
        test_learner.actor_network.set_weights(self.actor_network.get_weights())
        test_learner.critic_network.set_weights(self.critic_network.get_weights())
        return test_learner
