import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from abstract_brain import *



class neural_net(AbstractNeuralNetwork):
    def __init__(self, obs_space, action_space, network):
        self.obs_space = obs_space
        self.action_space = action_space
        self.model = Sequential()
        # initialise network object
        if network == 'dqn':
            neural_net.build_dqn(self)
        else if network == 'DDQN':
            neural_net.build_ddqn(self)

    def build_network(self):
        

        # 3 layers of convolutional networks
        # padding is added so that information is not loss when the kernal size is smaller
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation = 'relu', input_shape=self.obs_space, data_format='channels_last'))
        self.model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu', data_format='channels_last'))
        self.model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu', data_format='channels_last'))
        # convert image from 3D to 1D
        self.model.add(Flatten())

        # hidden layer takes a pre-processed frame as input, and has 200 units
        #  fibre channel layer 1
        self.model.add(Dense(units=200,input_dim=self.obs_space, activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        self.model.add(Dense(units=self.action_space, activation='sigmoid', kernel_initializer='RandomNormal'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print graph info
        # self.model.summary()
