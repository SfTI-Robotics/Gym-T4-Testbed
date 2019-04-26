import numpy as np

from agents.image_input import AbstractBrain
from agents.networks.policy_gradient_networks import build_actor_network, build_critic_network

action_dims = 1


class Learning(AbstractBrain.AbstractLearning):

    value_size = 1

    # create networks
    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)
        self.network = build_actor_network(self.state_space, self.action_space, self.config['learning_rate'])
        self.critic_network = build_critic_network(self.state_space, self.value_size, self.config['learning_rate'])

    def choose_action(self, state):
        # get policy from network
        policy = self.network.predict(np.array([state])).flatten()
        # pick action (stochastic)
        return np.random.choice(self.action_space, 1, p=policy)[0]

    def train_network(self, states, actions, rewards, next_states, dones, step):

        targets = np.zeros((self.config['batch_size'], self.value_size))
        advantages = np.zeros((self.config['batch_size'], self.action_space))

        values = self.critic_network.predict(states)
        next_values = self.critic_network.predict(next_states)

        for i in range(self.config['batch_size']):
            if dones[i]:
                advantages[i][actions[i]] = rewards[i] - values[i]
                targets[i][0] = rewards[i]
            else:
                advantages[i][actions[i]] = rewards[i] + self.config['gamma'] * (next_values[i]) - values[i]
                targets[i][0] = rewards[i] + self.config['gamma'] * next_values[i]

        self.network.fit(states, advantages, epochs=1, verbose=0)
        self.critic_network.fit(states, targets, epochs=1, verbose=0)


'''
    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        self.sess = tf.Session()
        K.set_session(self.sess)

        if config['environment'] == 'CartPole-v1':
            # init actor for cartpole
            self.actor_state_input, self.actor_model = \
                build_actor_cartpole_network(self.state_space, action_dims, self.config['learning_rate'])
            _, self.target_actor_model = \
                build_actor_cartpole_network(self.state_space, action_dims, self.config['learning_rate'])

            self.actor_critic_grad = tf.placeholder(tf.float32, [None, action_dims])

            # get gradient
            actor_model_weights = self.actor_model.trainable_weights
            self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
            grads = zip(self.actor_grads, actor_model_weights)
            self.optimize = tf.train.AdamOptimizer(self.config['learning_rate']).apply_gradients(grads)

            # init critic for cartpole
            self.critic_state_input, self.critic_action_input, self.critic_model = \
                build_critic_cartpole_network(self.state_space, action_dims, self.config['learning_rate'])
            _, _, self.target_critic_model = \
                build_critic_cartpole_network(self.state_space, action_dims, self.config['learning_rate'])

            self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

            # Initialize for later gradient calculations
            self.sess.run(tf.initialize_all_variables())

        # use network suitable for Atari games
        else:
            # init actor and critic for cartpole
            pass

    def _train_critic(self, states, actions, rewards, next_states, dones):
        # DQN-critic
        # TODO: try to fit entire batch
        for i in range(self.config['batch_size']):
            state = np.array([states[i]])
            next_state = np.array([next_states[i]])
            reward = rewards[i]
            action = actions[i]
            if not dones[i]:
                target_action = self.target_actor_model.predict(next_state)
                future_reward = self.target_critic_model.predict([next_state, target_action])[0][0]
                reward += self.config['gamma'] * future_reward
            a = state
            b = np.array([action])
            b = b.reshape((1, action_dims))
            c = np.array([reward])
            self.critic_model.fit([a, b], c, verbose=0)
            # self.critic_model.fit([cur_state, action], reward, verbose=0)

    def _train_actor(self, states):
        # policy gradient actor
        for i in range(self.config['batch_size']):
            state = np.array([states[i]])
            predicted_action = self.actor_model.predict(state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: state,
                self.critic_action_input: predicted_action
            })[0]
            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grad: grads
            })

    def _update_actor_target(self):
        self.target_actor_model.set_weights(self.actor_model.get_weights())

    def _update_critic_target(self):
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = random.randrange(self.action_space)
            if action < 0 or action > 1:
                print('WRONG ACTION')
        # TODO: Reduce dims of state?
        action = int(self.actor_model.predict(np.array([state])))
        if action < 0 or action > 1:
            print('WRONG ACTION')
        return action

    def update_epsilon(self):
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon = max(self.config['epsilon_min'], self.epsilon - self.epsilon_decay)

    def update_target_model(self):
        self._update_actor_target()
        self._update_critic_target()

    def train_network(self, states, actions, rewards, next_states, dones, step):
        if step % self.config['network_train_frequency'] == 0:
            self._train_critic(states, actions, rewards, next_states, dones)
            self._train_actor(states)
        if step % self.config['target_update_frequency'] == 0:
            print('# ========================================== UPDATE ========================================== #')
            self.update_target_model()
        self.update_epsilon()
'''
