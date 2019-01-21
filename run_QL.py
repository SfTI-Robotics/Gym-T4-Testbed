
import numpy as np
import gym
import os
import time


# parameters
EPISODE_NUM = 100
STEP_NUM = 40


"""
 inputs:
    GAME = gym environment name
    RL = algorithm object from python file

 """
def run_QL(env , RL):
# def run_QL(GAME , RL):
    
    # env = gym.make(GAME)

    

    for episode in range(EPISODE_NUM):

        episode_rewards = 0
        observation = env.reset()

        for step in range(STEP_NUM):

            os.system('clear')
            env.render()
            # time.sleep(0.1)

            action = RL.choose_action(env, observation)

            observation_, reward, done, info = env.step(action)

            RL.learn(observation, action, reward, observation_, done)

            episode_rewards += reward

            observation = observation_

            if done:


                break
                
    print('game over')
    env.close()

            







