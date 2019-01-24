import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
#import retro                 # Retro Environment


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import imageio
from skimage.transform import resize

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends


# WHAT I NEED
#processing
#cropping, resizing b and w and normalising
#stacking 4 frames
#dqn
#target and eval network + params
#running sessonb
#building network (conv 3 layers)
#storing transition
#learning
#need to megre target and behav every 5 iterations
#sampling a memory
#q target using bellman
#train eval network calculating cost
# choose action

EPISODE_NUM = 500
STEP_NUM = 100

env = gym.make('Pong-v0')
actions=env.action_space.n
states=env.observation_space

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())


class ProcessFrame:
    def __init__(self, frame_height=110, frame_width=84):
        self.frame_height=frame_height
        self.frame_width=frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8) # can input actions space and put it as the hspae try later
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed=self.proccesed/255
        self.processed = tf.image.resize_images(self.processed, 
                                                [self.frame_height, self.frame_width], 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def process(self, session, frame):
        return session.run(self.processed, feed_dict={self.frame:frame})

    stack_size = 4 # We stack 4 frames

    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

    def stack_frames(stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = process(state)
        
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2) 
        
        return stacked_state, stacked_frames
STACK_SIZE =4 

episode_rewards = 0
observation = env.reset()
for step in range(STACK_SIZE):
    observation, stacked_frames = stack_frames(observation, state, True)
    print("frames stacked")



# STATE_SIZE = [110,84,4]
# learning_rate = 0.00025
# BATCH_SIZE = 64

# EXPOLRATION_MAX=1.0
# EXPOLRATION_MIN=0.01
# DECAY_RATE= 0.00001

# GAMMA = 0.94

# PRETRAIN_LENGTH= BATCH_SIZE
# MEMORY_SIZE=1000000

# class DQNetwork:
#     def __init__(self, action_size, name='DQNetwork'):
        

#     with tf.variable_scope(name):
#         self.inputs_ = tf.placeholder(tf.float32, [None, *STATE_SIZE], name="inputs")
#         self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

#         self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
#                                          filters = 32,
#                                          kernel_size = [8,8],
#                                          strides = [4,4],
#                                          padding = "VALID",
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                          name = "conv1")
            
#          self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

#         self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
#                                  filters = 64,
#                                  kernel_size = [4,4],
#                                  strides = [2,2],
#                                  padding = "VALID",
#                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                  name = "conv2")

#         self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
        
#         self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
#                                  filters = 64,
#                                  kernel_size = [3,3],
#                                  strides = [2,2],
#                                  padding = "VALID",
#                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                  name = "conv3")
        
#         self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

#         self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
#         self.fc = tf.layers.dense(inputs = self.flatten,
#                                 units = 512,
#                                 activation = tf.nn.elu,
#                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                             name="fc1")
        
#         self.output = tf.layers.dense(inputs = self.fc, 
#                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                         units = self.action_size, 
#                                     activation=None)
        
#         # Q is our predicted Q value.
#         self.target_Q = tf.placeholder(tf.float32, [None], name="target")

#         self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
        
#         # The loss is the difference between our predicted Q_values and the Q_target
#         # Sum(Qtarget - Q)^2
#         self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        
#         self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#         self.update = self.optimizer.minimize(self.loss)
        
# class Memory():
#     def __init__(self, max_size):
#         self.buffer = deque(maxlen = max_size)

#     def add(self, experience):
#         self.buffer.append(experience)
    
#     def sample(self, batch_size):
#         buffer_size = len(self.buffer)
#         index = np.random.choice(np.arrange(buffer_size),
#                                 size = batch_size,
#                                 replace = False)
#         return [self.buffer[i] for i in index]

# def choose_action(self, state, episode, env):
#     epsilon = EXPOLRATION_MIN + (EXPOLRATION_MAX - EXPOLRATION_MIN) * np.exp(-DECAY_RATE * episode)

#     if(epsilon > np.random.rand())
#         action = env.action_space.sample()
#     else :
#         # exploitation: choose maximum action            
#         Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
#         # Take the biggest Q value (= the best action)
#         choice = np.argmax(Qs)
#         action = possible_actions[choice]
    
#     return action

# def learn():
#     if self.learn_step_counter % 5== 0:
#         self.sess.run(self.replace_target_op)
#         #only replace targets with evalutors after x amount of steps to give the learner new target
#         print('\ntarget_params_replaced\n')
    
