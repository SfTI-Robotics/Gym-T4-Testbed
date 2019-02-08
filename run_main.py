"""
this is the universal run script for all environments

"""
# print("enter -h for options you can enter")

import argparse
import sys
import numpy as np
import time
import gym

from argparse import RawTextHelpFormatter

from summary import summary
import time
import datetime
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-algorithm", help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
parser.add_argument("-environment", help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
# add global variables
parser.add_argument("-episodes", help="select number of episodes to graph")
# parser.add_argument("-steps", help="select number of steps")

args = parser.parse_args()

# Preprocessing folder
if args.environment == 'Pong-v0':
    import Preprocess.Pong_Preprocess as preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import Preprocess.SpaceInvaders_Preprocess as preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import Preprocess.MsPacman_Preprocess as preprocess
    print('MsPacman works')
else :

    print("Environment not found")

# # Brain and Network folder
if args.algorithm == 'QLearning':
    import Q_table.Brain as brain
    # import Q_table.Network as network
    print('Q tables work')
elif args.algorithm == 'DQN':
    import DQN.Brain as brain
    # import DQN.Network as network
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    import Double_DQN.Brain as brain
    # import Double_DQN.Network as network
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    import Dueling_DQN.Brain as brain
    # import Dueling_DQN.Network as network
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    import DDDQN_PER.Brain as brain
    # import DDDQN_PER.Network as network
    print('PER works')
else :
    print("Algorithm not found")
# ============================================
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_FILENAME = args.environment + '_' + args.algorithm
# ============================================

env = gym.make(args.environment)
action_space = env.action_space.n

# initialise objects
processor = preprocess.Processing()
state_space = processor.get_state_space()

print('state_space: ', env.observation_space.shape)

# neuralNet = network.neural_net(state_space, action_space)
# learner = brain.Learning(state_space, action_space, neuralNet)
learner = brain.Learning(state_space, action_space)

# if LOAD_MODEL == True:
#     neuralNet.model.save_weights(neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5'))

# Graphing results
now = datetime.datetime.now()
graph = summary(summary_types = ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_epsilon'],
            # the optimal step count of the optimal policy
            step_goal = 0,
            # the maximum reward for the optimal policy
            reward_goal = 0,
            # maximum exploitation value
            epsilon_goal = 0.99,
            # desired name for file
            NAME = MODEL_FILENAME + str(now),
            # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
            # SAVE_PATH = "/github/Gym-T4-Testbed/Gym-T4-Testbed/temp_Graphs/",
            SAVE_PATH = "/Gym-T4-Testbed/temp_Graphs/",
            EPISODE_MAX = int(args.episodes),

            STEP_MAX_M = processor.step_max,

            TIME_MAX_M = processor.time_max,

            REWARD_MIN_M = processor.reward_min,

            REWARD_MAX_M = processor.reward_max
    )
print("\n ==== initialisation complete, start training ==== \n")

# ==================================================
#
for episode in range(500):
    episode_transitions = []
    
    observation = env.reset()
    observation = processor.Preprocessing(observation, True)

    start_time = time.time()
    episode_rewards = 0
    step = 0
    game_step = 0
    while True:
        env.render()

        action= learner.choose_action(observation, episode)

        
        next_observation, reward, done, _ = env.step(action)

        ## Done s1 - s5 reward -1
        ## Iterate s5- s1 s5 -> -1, s4 -> %-1 ... etc

        next_observation = processor.Preprocessing(next_observation, False)
        episode_transitions.append((observation, action, reward, next_observation, done))
        # print(np.shape(episode_transitions))

        

        # if episode > 1:
        #     learner.memory_replay()
        game_step += 1  
        step += 1
        episode_rewards += reward
        episode_transitions = list(episode_transitions)

        if not reward == 0:
            for i in reversed(range(step)):
                print(episode_transitions[i][2])
                #  find reward of observations before it, change it based on a percentage
                episode_transitions[i][2] += reward * 0.2 ** (1/i)
        #call the memory replay function to learn at the end of every episode
        episode_transitions = tuple(episode_transitions)

        if done:
            print('Completed Episode ' + str(episode))
            print('reward =', episode_rewards, 'steps =', step)
            
            learner.transitions.extend(episode_transitions)
            learner.memory_replay()

            break

        observation = next_observation

    #observation -> [UP, DOWN, NO_OP]

    # store model weights and parameters when episode rewards are above a certain amount 
    # and after every number of episodes
    #  === change reward threshold
    
    # if (SAVE_MODEL == True and episode % 5 == 0):
    #     neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5', overwrite = True)

    graph.summarize(episode, step, time.time() - start_time, episode_rewards, learner.epsilon)

env.close()
