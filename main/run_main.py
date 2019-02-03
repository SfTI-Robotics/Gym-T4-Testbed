"""
this is the universal run script for all environments

"""
print("enter -h for options you can enter")
import argparse
import sys
import numpy as np
from argparse import RawTextHelpFormatter
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-algorithm", help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
parser.add_argument("-environment", help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
# add global variables
# parser.add_argument("-episodes", help="select number of episodes")
# parser.add_argument("-steps", help="select number of steps")

args = parser.parse_args()

# Preprocessing folder
if args.environment == 'Pong-v0':
    import Preprocess.Pong_Preprocess as preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import Preprocess.SpaceInvadersneural_net_Preprocess as preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import Preprocess.MsPacman_Preprocess as preprocess
    print('MsPacman works')
else :

    print("Environment not found")

# # Brain and Network folder
if args.algorithm == 'QLearning':
    import Q_table.Brain as brain
    import Q_table.Network as network
    print('Q tables work')
elif args.algorithm == 'DQN':
    import DQN.Brain as brain
    import DQN.Network as network
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    import Double_DQN.Brain as brain
    import Double_DQN.Network as network
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    import Dueling_DQN.Brain as brain
    import Dueling_DQN.Network as network
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    import DDDQN_PER.Brain as brain
    import DDDQN_PER.Network as network
    print('PER works')
else :
    print("Algorithm not found")

# ==================================================
# 
import gym

env = gym.make(args.environment)

action_space = env.action_space.n


# initialise objects
processor = preprocess.Processing()
state_space = processor.get_state_space()
# print('state=', state_space)
neuralNet = network.neural_net(state_space, action_space)
learner = brain.Learning(state_space, action_space, neuralNet)

print("initialisation complete, start training")

for episode in range(1000):
    observation = env.reset()

    observation  = processor.four_frames_to_state(observation, True)
    # print('OBS=', np.shape(observation))
    while True:
        env.render()
      
        print(observation.shape[:])
        action= learner.choose_action(observation, episode)

        next_observation, reward, done, _ = env.step(action)
        next_observation = processor.four_frames_to_state(next_observation, False)
        learner.transitions.append((observation, action, reward, next_observation, done))

        if done:
            print('Completed Episode ' + str(episode))
            #call the memory replay function to learn at the end of every episode
            learner.memory_replay()
            break

        observation = next_observation
