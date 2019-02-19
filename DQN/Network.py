import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
import numpy as np


class neural_net():
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        # action space uses the customised action encoding from the environment's preprocess file
        self.action_space = action_space

        self.model = Sequential()
        neural_net.build_network(self)

    def build_network(self):
        # 2 layers of convolutional networks
        # padding is added so that information is not loss when the kernal size is smaller
        self.model.add(Conv2D(16, kernel_size=(8, 8), strides = (2, 2), padding='valid', activation = 'relu', input_shape=self.obs_space, data_format='channels_first'))
        self.model.add(Conv2D(32, kernel_size=(4, 4), strides = (2, 2), padding='valid', activation = 'relu', data_format='channels_first'))
        # self.model.add(Conv2D(64, kernel_size=(3, 3), strides = (1, 1), padding='valid', activation = 'relu', data_format='channels_first'))

        # convert image from 2D to 1D
        self.model.add(Flatten())

        #layer for testing out code on cartpole
        #self.model.add(Dense(24, input_shape=(self.obs_space,), activation="relu"))

        # hidden layer takes a pre-processed frame as input, and has 200 units
        #  fibre channel layer 1
        self.model.add(Dense(units=500, activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        print("output layer dimensions = ", self.action_space)
        self.model.add(Dense(units=self.action_space, activation='softmax', kernel_initializer='RandomNormal'))

        # compile the model using traditional Machine Learning losses and optimizers
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # self.model.summary()
