"""
this is the universal run script for all environments

"""
# import dependencies
import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import time
import gym
# for graphing
from summary import summary
import time
import datetime
# recording environment render as video mp4
from gym.wrappers import Monitor
# allows gifs to be saved of the training episode for use in the Control Center.
import imageio

# ============================================

# For more on how argparse works see documentation
# create argument options
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-alg", "--algorithm", help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
parser.add_argument("-env","--environment", help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
parser.add_argument("-eps","--episodes", help="select number of episodes to graph")
# parser.add_argument("-steps", help="select number of steps")

# retrieve user inputted args from cmd line
args = parser.parse_args()

# Preprocessing folder
# this takes care of the environment specif7.8626167934090ics and image proccessing
if args.environment == 'Pong-v0':
    import Preprocess.Pong_Preprocess as preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import Preprocess.SpaceInvaders_Preprocess as preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import Preprocess.MsPacman_Preprocess as preprocess
    print('MsPacman works')
elif args.environment == 'Breakout-v0':
    import Preprocess.Breakout_Preprocess as preprocess
    print('Breakout works')
elif args.environment == 'CartPole-v1':
    import Preprocess.Cartpole_Preprocess as preprocess
    print('Cartpole works')
else :
    print("Environment not found")

# algorithm folder
# the newtork is imported into brain file in the header so no need to import the network here aswell
if args.algorithm == 'QLearning':
    import Q_table.Brain as brain
    print('Q tables work')
elif args.algorithm == 'DQN':
    import DQN.Brain as brain
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    import Double_DQN.Brain as brain
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    import Dueling_DQN.Brain as brain
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    import DDDQN_PER.Brain as brain
    print('PER works')
else :
    print("Algorithm not found")

# ============================================

# create gym env
env = gym.make(args.environment)
# initialise processing class specific to enviornment
processor = preprocess.Processing()
# state space is determined by the deque storing the frames from the env
state_space = processor.get_state_space()
if args.environment == 'Cartpole-v1':
    state_space = env.observation_space.shape[0]
    print("Goes into if loop")
# action space given by the environment
action_space = env.action_space.n

print(state_space)

print(action_space)

#**********************************************************************#
#if you want to look if there's any useless keys print the stuff below

# what_actions_do = env.unwrapped.get_action_meanings()
# print(what_actions_do)
#***********************************************************************#\

# here we change the action space if it contains 'useless' keys or actions that do the same thing
# if no useless keys it just returns the envs defined action space
# This function is created in the preprocess file
action_space=processor.new_action_space(action_space)
# initialise the algorithm class which also contains the network
learner = brain.Learning(state_space, action_space)

# ============================================

# Graphing results
now = datetime.datetime.now()
MODEL_FILENAME = args.environment + '_' + args.algorithm + '_'
# our graphing function
#summary sets the ranges and targets and saves the graph
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
            # episode upper bound for graph
            EPISODE_MAX = int(args.episodes),
            # step upper bound for graph
            STEP_MAX_M = processor.step_max,
            # time upper bound for graph
            TIME_MAX_M = processor.time_max,
            # reward upper bound for graph
            REWARD_MIN_M = processor.reward_min,
            # reward lower bound for graph
            REWARD_MAX_M = processor.reward_max
    )

# =================================================

DISCOUNTED_REWARDS_FACTOR=0.99
# ==================================================

# storing neural network weights and parameters
# SAVE_MODEL = True
# LOAD_MODEL = True
# if LOAD_MODEL == True:
#     neuralNet.model.save_weights(neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5'))

# ============================================
print("\n ==== initialisation complete, start training ==== \n")

for episode in range(int(args.episodes)):
    # storing frames as gifs, array emptied each new episode
    episode_frames = []

    observation = env.reset()
    episode_frames.append(observation)
    # Processing initial image cropping, grayscale, and stacking 4 of them
    observation = processor.Preprocessing(observation, True)

    start_time = time.time()
    episode_rewards = 0 #total rewards for graphing


    game_number = 0 # increases every time a someone scores a point
    game_step = 0 #for discounted rewards, steps for each round
    step=0 #count total steps for each episode for the graph

    # these arrays are used to calculated and store discounted rewards
    # arrays for other variable are needed for appending to transitions in our learner to work
    # arrays emptied after every round in an episode
    reward_array=[]
    reward_episode =[]
    states=[]
    actions=[]
    next_states=[]
    dones=[]

    while True:
        env.render()
        #action chooses from  simplified action space without useless keys
        action = learner.choose_action(observation, episode)
        # actions map the simp,ified action space to the environment action space
        # if action space has no useles keys then action = action_mapped
        action_mapped = processor.mapping_actions_to_keys(action)

        # takes a step
        next_observation, reward, done, _ = env.step(action_mapped)
        episode_frames.append(next_observation)

        # appending <s, a, r, s', d> into arrays for storage
        episode_rewards += reward
        reward_array.append(reward)
        states.append(observation)
        actions.append(action) # only append the '1 out of 3' action
        dones.append(done)
        next_observation = processor.Preprocessing(next_observation, False)
        next_states.append(next_observation)

        game_step += 1
        step+=1

        if (not reward == 0) or (done) :
            if args.environment == 'Pong-v0':
                # print(  'game_number =',   game_number , 'game_step = ', game_step)

                if reward > 0 :
                    # backpropagate the POSITIVE reward received so that the actions leading up to this result is accounted for
                    # philosophy of encouragement
                    reward_array=processor.discounted_rewards(reward_array,DISCOUNTED_REWARDS_FACTOR)

            # #append each <s, a, r, s', d> to learner.transitons for each game round
            # for i in range(game_step):
                # learner.transitions.append((states[i], actions[i], reward_array[i],next_states[i],dones[i]))
            reward_episode.append(reward_array)
            reward_array=[]
            game_number += 1
            game_step=0

            # when an agent's game score reaches 21
            if done:
                print('\n Completed Episode = ' + str(episode), 'steps = ', step, ' epsilon =', learner.epsilon, ' score = ', episode_rewards, '\n')
                # learner.network.model.fit(states, actions, sample_weight= reward_episode)


                # empty arrays after each round is complete
                states, actions, reward_episode, next_states, dones  = [], [], [], [], []
                # record video of environment render
                # env = gym.wrappers.Monitor(env,directory='Videos/' + MODEL_FILENAME + '/',video_callable=lambda episode_id: True, force=True,write_upon_reset=False)

                break


        observation = next_observation
        # train algorithm using experience replay
        learner.memory_replay()
    # make gif
    if episode != 0 and episode % 5 == 0:
        images = np.array(episode_frames)
        fname = './gifs/episode'+str(episode)+'.gif'
        with imageio.get_writer(fname, mode='I') as writer:
            for frame in images:
                writer.append_data(frame)


    # store model weights and parameters when episode rewards are above a certain amount
    # and after every number of episodes

    # if (SAVE_MODEL == True and episode % 5 == 0):
    #     neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5', overwrite = True)

    # summarize plots the graph
    graph.summarize(episode, step, time.time() - start_time, episode_rewards, learner.epsilon, learner.e_greedy_formula)
# killing environment to prevent memory leaks
env.close()
