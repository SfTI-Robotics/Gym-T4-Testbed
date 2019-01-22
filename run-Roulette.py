import gym
import numpy as np
from Roulette-v0 import QLearning
import os
import time

EPISODE_NUM = sys.argv[2]
STEP_NUM = sys.argv[3]

env = gym.make(sys.argv[1])
actions=env.action_space.n
states=env.observation_space.n
QLearning = QLearning(states, actions)



for episode in range(EPISODE_NUM):

    episode_rewards = 0
    observation = env.reset()

    for step in range(STEP_NUM):

        os.system('clear')
        time.sleep(0.1)

        action = QLearning.choose_action(observation,EPISODE_NUM)

        observation_, reward, done, _ = env.step(action)

        QLearning.learn(observation, action, reward, observation_, done)

        episode_rewards += reward

        observation = observation_

        if done:
                
            print("Episode Completed")
            break
        
            
print('game over')
env.close()

            







