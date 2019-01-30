import numpy as np
import gym
from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import random
from Pong_Network import *
from abstract_brain import *

MAX_MEMORY_LENGTH = 5000
LEARNING_RATE = 0.01
REWARD_DECAY = 0.9
START_TRAINING = 500
batch_size=10

class Processing(AbstractBrainPreProcess):
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)

    def Preprocessing(self, state):
        # grayscale
        frame = rgb2gray(state)
        frame=frame[35:195]              
        frame = frame / 255.0
        frame= transform.resize(frame,[80,80,1])

        return frame


    def four_frames_to_state(self, state, is_new_episode):
        
        frame = self.Preprocessing(state)
        # frame = self.Preprocessing(self, state)
        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)

        else:
            self.deque.append(frame)
            
        # reshape the deque
        stacked_state = np.stack(self.deque, axis = 0)

        return stacked_state




class Learning(AbstractBrainLearning):


    def __init__(self, actions):
        self.observation_space = (80, 80, 1)
        self.state_space = (4, 80, 80, 1)
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
            action = random.randrange(0, 4)#self.action_space
        else:
            action = np.argmax(self.net.model.predict(state))

        # decay epsilon
        self.epsilon = 0.01 + (0.99+0.01) * np.exp(-0.995 * episode)

        return action



    def memory_replay(self):
        # experience replay learning from our memories
        if len(self.transitions) < START_TRAINING:
            return

        # experience replay
        batch = random.sample(self.transitions, batch_size)

        print(type(batch))
#Option 1
###############################################################################################
        
        # initialise arrays
        states = np.zeros((batch_size, *self.state_space)) 
        next_states = np.zeros((batch_size, *self.state_space))
        action, reward, done = [], [], []
        target, target_next = [], []
        print('state:')
        print(np.shape(states))

        print('batch:')
        print(np.shape(batch))


        # extract variables from transition
        # extract seperate s,a,r.s'
        for i in range(batch_size):
            states[i] = batch[i][0]
            print('action: ')
            print(batch[i][1])
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_states[i] = batch[i][3]
            done.append(batch[i][4])  
            q = self.net.model.predict(states[i])
            print(q)
            target.append(q)
            target_next.append(self.net.model.predict(next_states[i]))
         # q target update
        
        print(np.shape(target))
        print('CHECK POINT')
        # target = self.net.model.predict(states)
        # target_next = self.net.model.predict(next_states)
###############################################################################################

#Option 2
###############################################################################################
        
        # states = np.array([each[0] for each in batch])
        # actions = np.array([each[1] for each in batch])
        # rewards = np.array([each[2] for each in batch])
        # next_states = np.array([each[3] for each in batch])
        # done_array = np.array([each[4] for each in batch])
        # q_target_array = []
        # next_q_value = 1

        # target = self.model.predict(states)
        # target_next = self.model.predict(next_states)
###############################################################################################
        for sample in range(batch_size):
            # check if transition was at end of episode
            is_done = done[sample]
            if is_done:
                target[sample][action[sample]] = reward[sample]
            else:
                # Bellman Equation
                target[sample][action[sample]] = reward[sample] + self.gamma * np.max(target_next[sample])

            self.net.model.fit(states[sample], target[sample], batch_size=batch_size,
            epochs=1, verbose=0)
        # calculates loss and does optimisation
        # run graph
        
