#the imports for adding command line arguments
import sys
import gym
import random
import time
from gym import wrappers

#setting up enumerators for the system arguments to run the code
ENVIRONMENT = sys.argv[1]
EPISODE = int(sys.argv[2])

#Coding for the two actions in Pong, declaration of the two potential actions that can be taken in Pong
UP_ACTION = 2
DOWN_ACTION = 3

#initializing the Pong Environment
env = gym.make(ENVIRONMENT)

#at the beginning of the episode you reset the environment, reset the environment in order to initialize itself.
observation = env.reset()                        


#Main loop UP_ACTION = 2
for i in range(EPISODE):
    
    #render the frame
    env.render()

    time.sleep(0.01)

    #choosing a random action to be performed
    action = random.randint(UP_ACTION, DOWN_ACTION)

    #Run a single step of the algorithm
    observation, reward, done, info = env.step(action)

    #if the episode is completed simply reset the environment
    if done:
        env.reset                 

env.close()
