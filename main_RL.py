import gym
import sys 
from run_QL import run_QL
# from Yonghui.test import brain_yonghui
from copy_QL import brain_QL
import numpy as np

# RL1 = brain_yonghui()
# print(RL1.print_one())
# RL2 = brain_kevin()

Game = 'Copy-v0'
env = gym.make(Game)

if hasattr(env.action_space, 'spaces'):
    dims = [d.n for d in env.action_space.spaces]
else:
    dims = [env.action_space.n]

actions=np.prod(dims)

states = env.observation_space.n
print(actions)
print(states)

# qVals = np.zeros((states, actions))
# print(qVals)

# for states in range(qVals[states,:]):
#     for actions in range(qVals[states,actions]):
#         print('(', states, ',', actions, ')' )

RL = brain_QL(actions, states)

run_QL(env, RL)
# run_QL(Game, RL2)

