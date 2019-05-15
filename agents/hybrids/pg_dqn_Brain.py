import datetime
import os
import sys

import agents.image_input.AbstractBrain as AbstractBrain
import agents.image_input.Double_DQN_Brain as QLearning
import agents.image_input.PPO_Brain as PolicyGradient


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        self.dqn_agent = QLearning.Learning(observations, actions, config)
        self.ppo_agent = PolicyGradient.Learning(observations, actions, config)

        self.step = 0
        self.switch = False
        if self.config['switch_steps'] == 0:
            self.switch = True

    def choose_action(self, state):
        # switch to q-learning
        if self.step == self.config['switch_steps']:
            print('# =========================================== SWITCH =========================================== #')
            self.switch = True
        self.step += 1

        if self.switch:
            return self.dqn_agent.choose_action(state)
        else:
            return self.ppo_agent.choose_action(state)

    def train_network(self, states, actions, rewards, next_states, dones, step):
        print('WARNING: training with train_network not possible, '
              'Policy Gradient and Deep Q-Learning need different training data '
              '(episode sequences vs random batch)')

    def train_pg_network(self, states, actions, rewards, next_states, dones, policies):
        self.ppo_agent.train_network(states, actions, rewards, next_states, dones, policies)

    def train_dqn_network(self, states, actions, rewards, next_states, dones, step):
        # store current value of epsilon, value will be updated during training
        temp_epsilon = self.dqn_agent.epsilon
        self.dqn_agent.train_network(states, actions, rewards, next_states, dones, step)
        # revert update of epsilon if switch has not happened yet
        if not self.switch:
            self.dqn_agent.epsilon = temp_epsilon

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path + 'dqn/'):
            os.makedirs(save_path + 'dqn/')
        if not os.path.exists(save_path + 'ppo/'):
            os.makedirs(save_path + 'ppo/')
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.dqn_agent.save_network(save_path + 'dqn/', model_name, timestamp=timestamp)
        self.ppo_agent.save_network(save_path + 'ppo/', model_name, timestamp=timestamp)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path + 'dqn/') and os.path.exists(save_path + 'ppo/'):
            self.dqn_agent.load_network(save_path + 'dqn/', model_name)
            self.ppo_agent.load_network(save_path + 'ppo/', model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")

    def get_test_learner(self):
        # get new agent for testing
        test_learner = Learning(self.state_space, self.action_space, self.config)
        # set weights of test networks to current weights
        test_learner.dqn_agent.behaviour_network.set_weights(self.dqn_agent.behaviour_network.get_weights())
        test_learner.dqn_agent.target_network.set_weights(self.dqn_agent.target_network.get_weights())
        test_learner.ppo_agent.actor_network.set_weights(self.ppo_agent.actor_network.get_weights())
        test_learner.ppo_agent.critic_network.set_weights(self.ppo_agent.critic_network.get_weights())
        return test_learner
