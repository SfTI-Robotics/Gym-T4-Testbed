import tensorflow as tf
import numpy as np
import gym
from collections import deque
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

# from neural_network import tensorflowNet



def StateProcessor(frames_deque, state, is_new_episode):
    # Preprocess frame
    
    gray = rgb2gray(state)
    
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12,4:-12]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    # Thanks to Mikołaj Walkowiak
    # 110x84x1 frame
    frame = transform.resize(normalized_frame, [110,84])

    # 4 frames stacked together as one state
    if is_new_episode:
        # Clear our frames_deque
        frames_deque = deque([np.zeros((110,84), dtype=np.int) for i in range(4)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        frames_deque.append(frame)
        frames_deque.append(frame)
        frames_deque.append(frame)with-deep-RL/blob/master/train.ipynb
        frames_deque.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(frames_deque, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        frames_deque.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(frames_deque, axis=2) 
    
    return stacked_state, frames_deque




    
class DeepQNetwork():
    def __init__(self, state_size, action_size):
        
        # initialise placeholders for state, action and q-target
        
        self.state = tf.placeholder(tf.float32, [None, *state_size], name = 'state')
        self.action = tf.placeholder(tf.float32, [None, action_size], name = 'action')
        self.q_target = tf.placeholder(tf.float32, [None], name = 'q_target')


        # pass frames through 3 convolutional neural networks (CNN or convnet)
        # (convolutional layers passed into an exponential or rectified linear unit network)
        self.conv1 = tf.nn.elu(tf.layers.conv2d(inputs = self.state,
                                filters = 32,
                                kernel_size = [8, 8],
                                strides = [4, 4],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )

        self.conv2 = tf.nn.elu(tf.layers.conv2d(inputs = self.conv1,
                                filters = 64,
                                kernel_size = [4, 4],
                                strides = [2, 2],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )

        self.conv3 = tf.nn.elu(tf.layers.conv2d(inputs = self.conv2,
                                filters = 64,
                                kernel_size = [2, 2],
                                strides = [1, 1],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )
        
        
        # flatten 4 frames to 1
        #  ----- not sure which one works
        self.flatten_frame = tf.contrib.layers.flatten(self.conv3)
        
        # pass frame data through 2 fibre channel (FC) layers
        self.fc1 = tf.layers.dense(inputs = self.flatten_frame,
                            units = 784,
                            activation = tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )

        self.fc2 = tf.layers.dense(inputs = self.fc1,
                            units = action_size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )

        # output a Q value for each action

        self.q_prediction = tf.reduce_sum(tf.multiply(self.fc2, self.action))

        self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_prediction))

        self.alpha = 0.8
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)



class trainAgent():

    def __init__(self):
        np.random.seed(3)
        self.epsilon = 1.0
        self.action_array = [0, 1, 2, 3, 4, 5]
        self.memory_capacity = 1000000 
        self.batch_size = 64
        self.gamma = 0.9

        env = gym.make('Pong-v0')
        state_size = [210, 160, 3]
        action_size = 6

        DQN = DeepQNetwork(state_size, action_size)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Initialize deque with zero-images one array for each image
        frames_deque  =  deque([np.zeros((110,84), dtype=np.int) for i in range(4)], maxlen=4)
        memory = deque(maxlen = self.memory_capacity)

        for episode in range(10000):

            state = env.reset()
            state, frames_deque = StateProcessor(frames_deque, state, True)
            
            episode_reward = 0

            for step in range(200):

                env.render()

                action = self.choose_action(state, episode)

                next_state, reward, done, _ = env.step(action)

                episode_reward += reward

                if not done: 
                    next_state, frames_deque = StateProcessor(frames_deque, next_state, False)

                    state = next_state

                    # store experience transition in memory
                    memory.append( (state, action, reward, next_state, done) )

                else: 
                    next_state = np.zeros(state.shape)
                    next_state, frames_deque = StateProcessor(frames_deque, next_state, False)

                    # store experience transition in memory
                    memory.append( (state, action, reward, next_state, done) )
                    
                    learn()

                    print('Episode ', episode, ' reward = ', episode_reward)
                    break

    def choose_action(self, state, episode):
        
        if np.random.rand() < self.epsilon:
            random_index = np.random.range(1, len(action_array)) - 1
            action = action_array[random_index]
        else:
            q_value = sess.run(DQN.fc2, feed_dict = {DQN.state: state.reshape((1, *state.shape))})
            action = action_array[np.argmax(q_value)]

        self.epsilon = 0.01 + 0.99 * np.exp(-0.995 * episode)
        return action


    def learn(self):
        # experience replay
        batch = memory.sample(batch_size)

        # extract variables from transition
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        done_array = np.array([each[4] for each in batch])
        q_target_array = []
        next_q_value = sess.run(DQN.fc2, feed_dict = {DQN.states: next_states})

        # update q_target
        for sample in range(batch_size):
            done = done_array[sample]

            if done:
                q_target_array.append(rewards[sample])
            else:
                q_target = rewards[sample] + self.gamma * np.max(next_q_value[sample])
                q_target_array.append(q_target)
        
        q_target = np.array([each for each in q_target_array])
        loss, optimizer = sess.run([DQN.loss, DQN.optimizer],
                                        feed_dict={DQN.state: states,
                                                   DQN.q_target: q_target,
                                                   DQN.actions: actions})

                      
    # def update_target(self, ):

if __name__ == "__main__":
    trainAgent = trainAgent()
        






