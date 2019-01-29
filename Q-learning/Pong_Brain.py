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
batch_size=80

                             
# initialise network




class Processing(AbstractBrainPreProcess):
    def __init__(self):
        self.deque = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)

    def Preproccesing(self, state):
        # grayscale
        frame = rgb2gray(state)   
        frame=frame[35:195]              #preprocesssing isn'      
        frame = frame / 255.0
        frame= transform.resize(frame,[80,80])                    



    def four_frames_to_state(self, state, is_new_episode):
        
        frame = self.Preprocessing(state)

        if is_new_episode:
            # all frames in new deque are of same state
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            self.deque.append(frame)
            
        else:
            self.deque.append(frame)
        # reshape the deque
        stacked_state = np.stack(self.deque, axis = 2)

        return stacked_state

    # def four_frames_to_nn():
        
                       

class Learning(AbstractBrainLearning):
    

    def __init__(self, actions):
        
        model = neural_net((80, 80, 1), actions)
        model.build_dqn()

        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.86
        #transitions is where we store memory of max memory length
        self.transitions = deque(maxlen = MAX_MEMORY_LENGTH)

    # the processed state is used in choosing action      
    def choose_action(self, state, episode):

        # e-greedy algorithm to choose actions
        if random.random() < self.epsilon: 
            action = 0
        else: 
            action = 0

        # decay epsilon
        self.epsilon = 0.01 + (0.99+0.01) * np.exp(-0.995 * episode)

        return action



    def memory_replay(self):
        # experience replay learning from our memories
        if len(self.transitions) < START_TRAINING:
            return

        # extract seperate s,a,r.s'
        # q target update
        # run graph

        # experience replay
        batch = random.sample(self.transitions, batch_size)


        update_input = np.zeros((BATCH_SIZE, self.observation_space))
        update_target = np.zeros((BATCH_SIZE, self.observation_space))
        action, reward, done = [], [], []

        # for i in range(BATCH_SIZE):
        #     update_input[i] = batch[i][0]
        #     action.append(batch[i][1])
        #     reward.append(batch[i][2])
        #     update_target[i] = batch[i][3]
        #     done.append(batch[i][4])

        
        # extract variables from transition
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        done_array = np.array([each[4] for each in batch])
        # q_target_array = []
        next_q_value = 1
        target = self.model.predict()

        for sample in range(batch_size):
            done = done_array[sample]
            if done:
                q_target_array.append(rewards[sample])
                self.model.fit()
            else:
                q_target = rewards[sample] + self.gamma * np.max(next_q_value[sample])
                q_target_array.append(q_target)

        # calculates loss and does optimisation 
        self.model.fit(state, target, batch_size=self.batch_size,
            epochs=1, verbose=0)

        


        






                    
                                
        

        

