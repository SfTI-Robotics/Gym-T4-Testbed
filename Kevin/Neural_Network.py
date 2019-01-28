import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

from obs_to_s import *


class neural_network:
    def __init__(
        self,
        state_num,
        action_num,
        ):
        self.obs_num = state_num
        self.action_num = action_num

    def build_conv_dqn(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape=(input_shape)))
        model.add(Conv2D(64, kernel_size=(5, 5), activation = 'relu'))
        model.add(Conv2D(64, kernel_size=(5, 5), activation = 'relu'))
        model.add(Flatten())
    
        # hidden layer takes a pre-processed frame as input, and has 200 units
        model.add(Dense(units=200,input_dim=input_shape, activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

        # compile the model using traditional Machine Learning losses and optimizers
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    