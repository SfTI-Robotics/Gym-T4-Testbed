import keras
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential


class NeuralNet:

    def __init__(self, obs_space, action_space, is_cartpole):
        self.obs_space = obs_space
        # action space uses the customised action encoding from the environment's preprocess file
        self.action_space = action_space

        self.model = Sequential()
        if is_cartpole:
            # this network works only for cartpole
            self.build_network22()
        else:
            # this network works for everything except cartpole
            self.build_network()

    def build_network(self):
        # 2 layers of convolutional networks
        # padding is added so that information is not loss when the kernel size is smaller
        self.model.add(Conv2D(16, kernel_size=(8, 8), strides=(2, 2), padding='valid', activation='relu',
                              input_shape=self.obs_space, data_format='channels_first'))
        self.model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                              data_format='channels_first'))

        # convert image from 2D to 1D
        self.model.add(Flatten())

        # layer for testing out code on cartpole
        # self.model.add(Dense(24, input_shape=(self.obs_space,), activation="relu"))

        # hidden layer takes a pre-processed frame as input, and has 200 units
        #  fibre channel layer 1
        self.model.add(Dense(units=500, activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        # print("output layer dimensions = ", self.action_space)
        self.model.add(Dense(units=self.action_space, activation='softmax', kernel_initializer='RandomNormal'))

        # compile the self.model using traditional Machine Learning losses and optimizers
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # self.model.summary()

    def build_network22(self):
        self.model.add(Dense(24, input_shape=(self.obs_space,), activation="relu"))
        # self.model.add(Dense(512, activation='relu', kernel_initializer='he_uniform' ))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.00025), loss='mse')
        # self.model.summary()

    def build_network_double_dqn(self):
        # 3 layers of convolutional networks
        # padding is added so that information is not loss when the kernel size is smaller
        self.model.add(
            Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=self.obs_space,
                   data_format='channels_first'))
        self.model.add(
            Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', data_format='channels_first'))
        self.model.add(
            Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', data_format='channels_first'))
        # convert image from 3D to 1D
        self.model.add(Flatten())

        # hidden layer takes a pre-processed frame as input, and has 200 units
        #  fibre channel layer 1
        self.model.add(
            Dense(units=200, input_dim=self.obs_space, activation='relu', kernel_initializer='he_uniform'))

        # output layer
        self.model.add(Dense(units=self.action_space, activation='relu', kernel_initializer='RandomNormal'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # self.model.summary()

    def build_network_dueling_dqn(self):
        self.model.add(
            Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=self.obs_space,
                   data_format='channels_first'))
        self.model.add(
            Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', data_format='channels_first'))
        self.model.add(
            Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(
            Dense(units=512, input_dim=self.obs_space, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(units=self.action_space, activation='relu', kernel_initializer='RandomNormal'))

        advantage = Dense(units=self.action_space, activation='relu', kernel_initializer='RandomNormal')
        value = Dense(units=1, activation='relu', kernel_initializer='RandomNormal')

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        advantage = Dense(NUM_ACTIONS)(fc1)

        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = merge([advantage, value], mode=lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(NUM_ACTIONS,))
        # policy = Dense(NUM_ACTIONS)(merge_layer)

        self.model = Model(input=[input_layer], output=[policy])
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = Model(input=[input_layer], output=[policy])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))
