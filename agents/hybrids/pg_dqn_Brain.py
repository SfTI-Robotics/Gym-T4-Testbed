import datetime
import os
import sys

import agents.image_input.AbstractBrain as AbstractBrain
import agents.image_input.DQN_Brain as DQN
import agents.image_input.Double_DQN_Brain as DoubleDQN
import agents.image_input.Dueling_Brain as DuelingDQN
import agents.image_input.Policy_Gradient_Brain as PG


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        # self.dqn_agent = DQN.Learning(observations, actions, config)
        # self.dqn_agent = DoubleDQN.Learning(observations, actions, config)
        self.dqn_agent = DuelingDQN.Learning(observations, actions, config)
        self.pg_agent = PG.Learning(observations, actions, config)

        self.step = 0
        self.switch = False
        if self.config['switch_steps'] == 0:
            self.switch = True

    def choose_action(self, state, print_predictions=False):
        # switch to q-learning
        if self.step == self.config['switch_steps']:
            print('# =========================================== SWITCH =========================================== #')
            self.switch = True
        self.step += 1

        if self.switch:
            return self.dqn_agent.choose_action(state, print_predictions=print_predictions)
        else:
            return self.pg_agent.choose_action(state, print_predictions=print_predictions)

    def train_network(self, states, actions, rewards, next_states, dones, step):
        print('WARNING: training with train_network not possible, '
              'Policy Gradient and Deep Q-Learning need different training data '
              '(episode sequences vs random batch)')

    def train_pg_network(self, states, actions, rewards, next_states, dones, step):
        self.pg_agent.train_network(states, actions, rewards, next_states, dones, step)

    def train_dqn_network(self, states, actions, rewards, next_states, dones, step):
        temp_epsilon = self.dqn_agent.epsilon

        self.dqn_agent.train_network(states, actions, rewards, next_states, dones, step)

        if not self.switch:
            self.dqn_agent.epsilon = temp_epsilon

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path + 'dqn/'):
            os.makedirs(save_path + 'dqn/')
        if not os.path.exists(save_path + 'pg/'):
            os.makedirs(save_path + 'pg/')
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.dqn_agent.save_network(save_path + 'dqn/', model_name, timestamp=timestamp)
        self.pg_agent.save_network(save_path + 'pg/', model_name, timestamp=timestamp)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path + 'dqn/') and os.path.exists(save_path + 'pg/'):
            self.dqn_agent.load_network(save_path + 'dqn/', model_name)
            self.pg_agent.load_network(save_path + 'pg/', model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")
