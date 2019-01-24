import tensorflow as tf
import numpy as np
# from neural_network import tensorflowNet



class DQN:
    def __init__(self, ):

        # 1. initialise instance variables
        self.alpha = 0.85
        self.gamma = 0.98
        self.epsilon = 1.0


        # 2. call build net function
        # 3. get parameters

        # 4. instantiate session
        sess = tf.Session()

        # 5. launch session
        sess.run(tf.global_variables_initializer)

        
        # preprocess frames

        greyscale = rgb2gray(frame)

        

        

        
        # initialise placeholders for state, action and q-target
        
        state = tf.placeholder(tf.float32, [None, state_size], name = 'state')
        action = tf.placeholder(tf.float32, [None, action_size], name = 'action')
        q_target = tf.placeholder(tf.float32, [None], name = 'q_target')

        # take a stack of 4 frames as input
        



        # pass frames through 3 convolutional neural networks (CNN or convnet)
        # (convolutional layers passed into an exponential or rectified linear unit network)
        conv1 = tf.nn.elu(tf.layers.con2d(inputs = state,
                                filters = 32,
                                kernel_size = [8, 8],
                                strides = [4, 4],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )

        conv2 = tf.nn.elu(tf.layers.con2d(inputs = conv1,
                                filters = 64,
                                kernel_size = [4, 4],
                                strides = [2, 2],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )

        conv3 = tf.nn.elu(tf.layers.con2d(inputs = conv2,
                                filters = 64,
                                kernel_size = [2, 2],
                                strides = [1, 1],
                                padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
        )
        
        
        # flatten 4 frames to 1
        #  ----- not sure which one works
        flatten_frame = tf.contrib.layers.flatten(conv3)
        
        # pass frame data through 2 fibre channel (FC) layers
        fc1 = tf.layers.dense(inputs = flatten_frame,
                            units = 784,
                            activation = tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )

        fc2 = tf.layers.dense(inputs = fc1,
                            units = action_size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )

        # output a Q value for each action

        q_prediction = tf.reduce_sum(tf.multiply(fc2, action))

        loss = tf. 
        

    








    def choose_action(self):


    def learn(self):


    def update_target(self, ):




