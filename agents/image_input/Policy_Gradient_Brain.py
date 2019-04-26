import numpy as np
from numpy.core.tests.test_mem_overlap import xrange

from agents.image_input import AbstractBrain
from agents.networks.policy_gradient_networks import build_policy_gradient_cartpole_network


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)
        if config['environment'] == 'CartPole-v1':
            # init actor and critic for cartpole
            self.network = build_policy_gradient_cartpole_network(self.state_space, self.action_space,
                                                                  self.config['learning_rate'])
        # use network suitable for Atari games
        else:
            # init actor and critic for cartpole
            print('Atari environments not implemented for Policy Gradients')
            exit(42)
            pass

    def discount_rewards(self, rewards):
        """ takes 1D float array of rewards and computes discounted reward """
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(xrange(0, rewards.size)):
            running_add = running_add * self.config['gamma'] + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def choose_action(self, state):
        predict = self.network.predict(state)[0]
        action = np.random.choice(self.action_space, 1, p=predict.reshape(self.action_space))[0]

        predict2 = self.network.predict([state])[0]
        action2 = np.argmax(predict)
        return action

    def train_network(self, states, actions, rewards):
        discounted_rewards = np.empty(0).reshape(0, 1)
        # Determine standardized rewards
        discounted_rewards_episode = self.discount_rewards(rewards)
        discounted_rewards = np.vstack([discounted_rewards, discounted_rewards_episode])

        rewards = np.empty(0).reshape(0, 1)

        # used if multiple episodes are put into one training-batch
        # if (num_episode + 1) % batch_size == 0:
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()
        discounted_rewards = discounted_rewards.squeeze()
        actions = actions.squeeze().astype(int)

        actions_train = np.zeros([len(actions), self.action_space])
        actions_train[np.arange(len(actions)), actions] = 1

        loss = self.network.train_on_batch([states, discounted_rewards], actions_train)
