
import numpy as np
import gym
import os
import time


# parameters
EPISODE_NUM = 5000
STEP_NUM = 100


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
            time.sleep(0.1)

            #action = RL.choose_action(env, observation)
            action=(1,1, observation)

            observation_, reward, done, _ = env.step(action)

            RL.learn(observation, action, reward, observation_)

            episode_rewards += reward

            observation = observation_

            if done:
                    
                print("Episode Completed")
                break
           
                
    print('game over')
    env.close()

            







