

class Learning(AbstractBrainLearning):


    def __init__(self, actions):
        self.observation_space = (80, 80, 4)
        self.state_space = (80, 80, 4)
        self.action_space = actions

        self.net = neural_net(self.observation_space, self.action_space)
        
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.86
        #transitions is where we store memory of max memory length
        self.transitions = deque(maxlen = MAX_MEMORY_LENGTH)

    # the processed state is used in choosing action
    def choose_action(self, state, episode):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_space)#self.action_space
        else:
            action = np.argmax(self.net.model.predict(np.expand_dims(state, axis = 0)))

        # decay epsilon
        self.epsilon = 0.01 + (0.99+0.01) * np.exp(-0.995 * episode)

        return action



    def memory_replay(self):
        # experience replay learning from our memories
        if len(self.transitions) < START_TRAINING:
            return

        # experience replay
        batch = random.sample(self.transitions, batch_size)
###############################################################################################
        
        # initialise arrays
        states = np.zeros((batch_size, *self.state_space)) 
        next_states = np.zeros((batch_size, *self.state_space))
        action, reward, done = [], [], []
        


        # extract variables from transition
        # extract seperate s,a,r.s'
        for i in range(batch_size):
            states[i] = np.array(batch[i][0])
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = np.array(batch[i][3])
            done.append(batch[i][4])  

        target = self.net.model.predict(states, batch_size=batch_size)
        target_next = self.net.model.predict(next_states, batch_size=batch_size)
###############################################################################################

        for sample in range(batch_size):
            # check if transition was at end of episode
            is_done = done[sample]
            if is_done:
                target[sample][action[sample]] = reward[sample]
            else:
                # Bellman Equation
                target[sample][action[sample]] = reward[sample] + self.gamma * np.max(target_next[sample])

            
        # print(target.shape[:])
        # calculates loss and does optimisation
        # run graph
        self.net.model.fit(states, target, batch_size=batch_size,
        epochs=1, verbose=0)