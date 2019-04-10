from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def build_dqn_network(obs_space, action_space, learning_rate):
    model = Sequential()
    # 2 layers of convolutional networks
    # padding is added so that information is not loss when the kernel size is smaller
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation='relu',
                     input_shape=obs_space, data_format='channels_first'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu',))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',))
    # convert image from 2D to 1D
    model.add(Flatten())

    # hidden layer takes a pre-processed frame as input, and has 200 units
    #  fibre channel layer 1
    model.add(Dense(units=512, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    model.add(Dense(units=action_space, activation='softmax', kernel_initializer='RandomNormal'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    # self.model.summary()
    return model


def build_cartpole_network(obs_space, action_space, learning_rate):
    model = Sequential()
    model.add(Dense(24, input_shape=obs_space, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_space, activation='linear', kernel_initializer='he_uniform'))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    # self.model.summary()
    return model
