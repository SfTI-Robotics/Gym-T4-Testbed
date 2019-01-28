from Neural_Network import *
import obs_to_s as pp

import numpy as np
import gym

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

print(type(env.observation_space))
print(env.observation_space)
obs = env.observation_space.shape[:]
print(obs)



nn = neural_network(env.observation_space.shape[0], env.action_space.shape[0])
nn.build_conv_dqn(80*80)
# Macros
UP_ACTION = 2
DOWN_ACTION = 3

# Hyperparameters
gamma = 0.99

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0

# main loop
while (True):

    # preprocess the observation, set input as difference between images
    cur_input = pp.pong_preprocess(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input
    
    # forward the policy network and sample action according to the proba distribution
    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0 # 0 and 1 are our labels

    # log the input and label to train later
    x_train.append(x)
    y_train.append(y)

    # do one step in our environment
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward
    
    # end of an episode
    if done:
        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        
        # increment episode number
        episode_nb += 1
        
        # training
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards, gamma))
                                                             
        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None