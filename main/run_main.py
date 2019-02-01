"""
this is the universal run script for all environments

"""
print("enter -h for options you can enter")
import argparse
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
    import Preprocess.Pong_Preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import Preprocess.SpaceInvaders_Preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import Preprocess.MsPacman_Preprocess
    print('MsPacman works')
else :

    print("Environment not found")

# # Brain and Network folder
if args.algorithm == 'QLearning':
    import Q_table.Brain
    import Q_table.Network
    print('Q tables work')
elif args.algorithm == 'DQN':
    import DQN.Brain
    import DQN.Network
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    import Double_DQN.Brain
    import Double_DQN.Network
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    import Dueling_DQN.Brain
    import Dueling_DQN.Network
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    import DDDQN_PER.Brain
    import DDDQN_PER.Network
    print('PER works')
else :
    print("Algorithm not found")