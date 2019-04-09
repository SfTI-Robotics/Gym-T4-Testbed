import keras
from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, K
from keras.models import Sequential


# TODO: networks depend on algorithm -> restructure
from keras.optimizers import Adam


class NeuralNetworkBuilder:

    @staticmethod
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
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        # self.model.summary()
        return model

    # TODO: make this work
    @staticmethod
    def build_dueling_dqn_network(obs_space, action_space, learning_rate):
        # see https://github.com/UoA-RL/Gym-T4-Testbed/blob/henry_test/models.py
        state_input = Input(shape=obs_space)
        x = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                   data_format='channels_first')(state_input)
        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, activation='relu')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_space,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu')(x)
        action_advantage = Dense(action_space)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                  output_shape=(action_space,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = keras.layers.add([state_value, action_advantage])

        model = Model(input=state_input, output=state_action_value)
        # model.compile(rmsprop(lr=learning_rate), "mse")
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        # model.summary()
        return model

    # TODO: try to use dqn network for cartpole
    @staticmethod
    def build_cartpole_network(obs_space, action_space, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_shape=obs_space, activation="relu", kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(action_space, activation='linear', kernel_initializer='he_uniform'))
        # TODO: extract learning rate
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        # self.model.summary()
        return model

    @staticmethod
    def build_dueling_cartpole_network(obs_space, action_space, learning_rate):
        state_input = Input(shape=obs_space)
        x = Dense(24, activation='relu')(state_input)
        x = Dense(24, activation='relu')(x)

        state_value = Dense(12, activation='relu')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(12, activation='relu')(x)
        action_advantage = Dense(action_space)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                  output_shape=(action_space,))(action_advantage)

        state_action_value = keras.layers.add([state_value, action_advantage])
        model = Model(input=state_input, output=state_action_value)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        # model.summary()
        return model
