"""
BipedalWalker-v2 

State:
- hull angle speed
- angular velocity 
- horizontal & vertical speed
- position of joints 
- joint angular speed
- number of legs contact with ground
- 10 lidar rangefinder measurements


Reward: 
- moving forward = 300
- falling = -100




"""


from keras import *
import numpy
import gym

def run_BipedalWalker_v2():

    env = gym.make('BipedalWalker-v2')

    state = env.reset()

