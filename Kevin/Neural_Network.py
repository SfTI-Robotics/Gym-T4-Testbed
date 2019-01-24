import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

class neural_network:
    def __init__(
        self,
        state_num,
        action_num,
        ):
        self.obs_num = state_num
        self.action_num = action_num


    def preporcess(image):
        image = image[35:195]
        image = image[::2, ::2, 0]
        image[image == 144] = 0
        image[image == 109] = 0
        image[image != 0] = 1
        
        return image.astype(np.float).ravel()


    def build_conv_net(input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape=(input_shape))
        model.add(Conv2D(64, kernel_size=(5, 5), activation = 'relu'))
        model.add(Conv2D(64, kernel_size=(5, 5), activation = 'relu'))
        model.add(Flatten())
        
