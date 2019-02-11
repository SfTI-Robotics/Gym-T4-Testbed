def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards
    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate
    Returns:
        discounted_rewards: same shape as `rewards`
    Notes:
        In Pong, when the reward can be {-1, 0, 1}.
        However, when the reward is either -1 or 1,
        it means the game has been reset.
        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r

def train(self, states, actions, rewards):
    states = np.array(states)
    actions = np.array(actions) - 1
    rewards = np.array(rewards)

    feed = {
        self.local.states: states
    }

    values = self.sess.run(self.local.values, feed)

    rewards = discount_reward(rewards, gamma=0.99)
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)

    advantage = rewards - values
    advantage -= np.mean(advantage)
    advantage /= np.std(advantage) + 1e-8

    feed = {
        self.local.states: states,
        self.local.actions: actions,
        self.local.rewards: rewards,
        self.local.advantage: advantage
    }

def play_episode(self):
    self.sess.run(self.global_to_local)

    states = []
    actions = []
    rewards = []

    s = self.env.reset()
    s = pipeline(s)
    state_diff = s

    done = False
    total_reward = 0
    time_step = 0
    while not done:

        a = self.choose_action(state_diff)
        s2, r, done, _ = self.env.step(a)

        s2 = pipeline(s2)
        total_reward += r

        states.append(state_diff)
        actions.append(a)
        rewards.append(r)

        state_diff = s2 - s
        s = s2

        if r == -1 or r == 1 or done:
            time_step += 1

            if time_step >= 5 or done:
                self.train(states, actions, rewards)
                self.sess.run(self.global_to_local)
                states, actions, rewards = [], [], []
                time_step = 0

    self.print(total_reward)
def run(self):
    while not self.coord.should_stop():
        self.play_episode()