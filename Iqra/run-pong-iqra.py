import numpy as np           # Handle matrices
import retro                 # Retro Environment
import gym
from pong-v0-iqra import DQNetwork
from pong-v0-iqra import ProcessFrame
from pong-v0-iqra import Memory()

import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

EPISODE_NUM = int(sys.argv[2])
STEP_NUM = int(sys.argv[3])

env = gym.make(sys.argv[1])

actions=env.action_space.n
states=env.observation_space
print(actions)
print(states)

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
print(possible_actions)
RL = QLearning(states, actions)


for episode in range(EPISODE_NUM):

    episode_rewards = 0
    observation = env.reset()

    start_time = time.time()

    for step in range(STEP_NUM):

        # os.system('clear')
        # time.sleep(0.1)
        env.render()

        # roulete doesn't have env.render
        observation=stack_frames(observation)

        #each step you choose action by calling the choose action function in our algorithm file
        action = RL.choose_action(observation,EPISODE_NUM, env)

        # apply action
        observation_, reward, done, _ = env.step(action)

        #learn from the action(updating q values)
        RL.learn(observation, action, reward, observation_, done)
        #get cumalative reward
        episode_rewards += reward
        #move on to next state
        observation = observation_

        if done:
            print('Episode =', episode, ',  reward =', episode_rewards)
            break
    
        
            
print('game over')
env.close()
