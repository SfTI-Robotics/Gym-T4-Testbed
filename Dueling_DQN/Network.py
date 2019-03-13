import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


class neural_net():
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

        self.model = Sequential()
        # initialise network object
        neural_net.build_network(self)

    def build_network(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation = 'relu', input_shape=self.obs_space, data_format='channels_first'))
        self.model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu', data_format='channels_first'))
        self.model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation = 'relu', data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512,input_dim=self.obs_space, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(units=self.action_space, activation='relu', kernel_initializer='RandomNormal'))

        advantage = Dense(units=self.action_space, activation='relu', kernel_initializer='RandomNormal')
        value = Dense(units=1, activation='relu', kernel_initializer='RandomNormal')

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
       
        
        advantage = Dense(NUM_ACTIONS)(fc1)

        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (NUM_ACTIONS,))
        # policy = Dense(NUM_ACTIONS)(merge_layer)

        self.model = Model(input=[input_layer], output=[policy])
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = Model(input=[input_layer], output=[policy])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        