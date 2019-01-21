
import numpy as np
import gym



# parameters
EPISODE_NUM = 
STEP_NUM =


"""
 inputs:
    GAME = gym environment name
    RL = algorithm object from python file

 """

def run_QL(GAME , RL):
    
    env = gym.make(GAME)

    

    for episode in range(EPISODE_NUM):

        episode_rewards = 0
        observation = env.reset()

        for step in range(STEP_NUM):

            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.learn(observation, action, reward, observation_ )

            episode_rewards += reward

            observation = observation_

            if done:


                break
                
    print('game over')
    env.destroy()

            







