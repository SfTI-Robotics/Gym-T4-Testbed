import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
import agents.image_input.DQN_Brain as DQN
import agents.image_input.Policy_Gradient_Brain as PG


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        self.dqn_agent = DQN.Learning(observations, actions, config)
        self.pg_agent = PG.Learning(observations, actions, config)

        self.switch = False
        self.train_counter = 0
        self.switch_point = 100

        # make sure DQN actually trains in ever
        self.config['network_train_frequency'] = 1
        self.config['target_update_frequency'] = 1

    def choose_action(self, state):
        if self.switch:
            return self.dqn_agent.choose_action(state)
        else:
            return self.pg_agent.choose_action(state)

    def train_network(self, states, actions, rewards, next_states, dones, step):

        if self.train_counter == self.config['switch_steps']:
            print('# =========================================== SWITCH =========================================== #')
            self.switch = True

        if not self.switch:
            # only train policy gradient before switch
            self.pg_agent.train_network(states, actions, rewards, next_states, dones, step)

        # always train dqn
        self.dqn_agent.train_hybrid_network(states, actions, rewards, next_states, dones, step, self.switch)

        self.train_counter += 1

    def update_weights(self):
        self.dqn_agent.network.set_weights(self.pg_agent.network.get_weights())
