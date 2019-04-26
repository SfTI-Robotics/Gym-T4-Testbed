from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_actor_network(obs_space, action_space, learning_rate):
    actor = Sequential()
    actor.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation='relu',
                     input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    actor.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                     input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    actor.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    # convert image from 2D to 1D
    actor.add(Flatten())

    actor.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform'))

    # output layer
    actor.add(Dense(units=action_space, activation='softmax', kernel_initializer='he_uniform'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    actor.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate))
    # self.model.summary()
    return actor


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_critic_network(obs_space, value_size, learning_rate):
    critic = Sequential()
    critic.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    critic.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    critic.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    # convert image from 2D to 1D
    critic.add(Flatten())

    critic.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform'))

    # output layer
    critic.add(Dense(units=value_size, activation='softmax', kernel_initializer='he_uniform'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    critic.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    # self.model.summary()
    return critic


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_actor_cartpole_network(obs_space, action_space, learning_rate):
    actor = Sequential()
    actor.add(Dense(24, input_dim=obs_space[0], activation='relu',
                    kernel_initializer='he_uniform'))
    actor.add(Dense(action_space, activation='softmax',
                    kernel_initializer='he_uniform'))
    actor.summary()
    # See note regarding crossentropy in cartpole_reinforce.py
    actor.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate))
    return actor


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_critic_cartpole_network(obs_space, value_size, learning_rate):
    critic = Sequential()
    critic.add(Dense(24, input_dim=obs_space[0], activation='relu',
                     kernel_initializer='he_uniform'))
    critic.add(Dense(value_size, activation='linear',
                     kernel_initializer='he_uniform'))
    critic.summary()
    critic.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return critic


'''
# see https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
def build_actor_cartpole_network(obs_space, action_space, learning_rate):
    state_input = Input(shape=obs_space)
    h1 = Dense(24, activation='relu')(state_input)
    h2 = Dense(48, activation='relu')(h1)
    h3 = Dense(24, activation='relu')(h2)
    output = Dense(action_space, activation='relu')(h3)

    model = Model(input=state_input, output=output)
    adam = Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=adam)
    # return reference to input layer together with model
    return state_input, model


# see https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
def build_critic_cartpole_network(obs_space, action_space, learning_rate):
    state_input = Input(shape=obs_space)
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(48)(state_h1)

    action_input = Input(shape=(action_space,))
    action_h1 = Dense(48)(action_input)

    merged = Add()([state_h2, action_h1])
    merged_h1 = Dense(24, activation='relu')(merged)
    output = Dense(1, activation='relu')(merged_h1)
    model = Model(input=[state_input, action_input],
                  output=output)

    adam = Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=adam)
    # return reference to input layers (used to update actor) together with model
    return state_input, action_input, model
'''
