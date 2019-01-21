import gym
import sys 
from run_QL import run_QL
from Yonghui.test import brain_yonghui


RL1 = brain_yonghui()
# print(RL1.print_one())
# RL2 = brain_kevin()

Game = 'Copy-v0'

run_QL(Game, RL1)
# run_QL(Game, RL2)