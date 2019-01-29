import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D



class neural_net():
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

        self.build_dqn()

    # https://github.com/mtrazzi/spinning-up-a-Pong-AI-with-deep-RL/blob/master/train.ipynb
    # https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
    def build_dqn(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation = 'relu', input_shape=self.obs_space))
        model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu'))
        model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu'))
        model.add(Flatten())
    
        # hidden layer takes a pre-processed frame as input, and has 200 units
        model.add(Dense(units=200,input_dim=self.obs_space, activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

        # compile the model using traditional Machine Learning losses and optimizers
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print graph info
        model.summary()

     