import agents.image_input.AbstractBrain as AbstractBrain
import agents.image_input.DQN_Brain as DQN
import agents.image_input.Policy_Gradient_Brain as PG


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        self.dqn_agent = DQN.Learning(observations, actions, config)
        self.pg_agent = PG.Learning(observations, actions, config)

        if self.config['switch_steps'] == 0:
            self.switch = True
        else:
            self.switch = False
        self.switch_counter = 0

    def choose_action(self, state):
        if self.switch:
            return self.dqn_agent.choose_action(state)
        else:
            return self.pg_agent.choose_action(state)

    def train_network(self, states, actions, rewards, next_states, dones, step):
        print('WARNING: training with train_network not possible, '
              'Policy Gradient and Deep Q-Learning need different training data '
              '(episode sequences vs random batch)')

    def train_pg_network(self, states, actions, rewards, next_states, dones, step):
        self.pg_agent.train_network(states, actions, rewards, next_states, dones, step)
        # switch to q-learning
        self.switch_counter += 1
        if self.switch_counter == self.config['switch_steps']:
            print('# =========================================== SWITCH =========================================== #')
            self.switch = True

    def train_dqn_network(self, states, actions, rewards, next_states, dones, step):
        temp_epsilon = self.dqn_agent.epsilon

        self.dqn_agent.train_network(states, actions, rewards, next_states, dones, step)

        if not self.switch:
            self.dqn_agent.epsilon = temp_epsilon
