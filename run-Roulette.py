
import gym
import numpy as np
# from /home/lemon740/Gym-T4-Testbed import Roulette
#this is importing the algorithm which is defined as a class from a diffeerent folder
from Roulette import QLearning
import os
import time
import sys

#our bash file takes the arg.txt file that has a list of the different environments episodes and steps
#when using sys.argv start by using 1 not 0 for the first agrument because 0 will be the python file name
EPISODE_NUM = int(sys.argv[2])
STEP_NUM = int(sys.argv[3])

env = gym.make(sys.argv[1])
#action space=38
#state =1
actions=env.action_space.n
states=env.observation_space.n
# the class doesn't take in any parameters but to pass in the action and state space in your class
#  use the name __init__ with double underscores
QLearning = QLearning(states, actions)

for episode in range(EPISODE_NUM):

    episode_rewards = 0
    observation = env.reset()

    for step in range(STEP_NUM):

        # os.system('clear')
        # time.sleep(0.1)

        # roulete doesn't have env.render

        #each step you choose action by calling the choose action function in our algorithm file
        action = QLearning.choose_action(observation,EPISODE_NUM, env)

        # apply action
        observation_, reward, done, _ = env.step(action)

        #learn from the action(updating q values)
        QLearning.learn(observation, action, reward, observation_, done)
        #get cumalative reward
        episode_rewards += reward
        #move on to next state
        observation = observation_

        if done:
            print('Episode =', episode, ',  reward =', episode_rewards)
                
            print("Episode Completed")
            break
        
            
print('game over')
env.close()

            







