"""
this is the universal run script for all environments

"""
# import dependencies
import argparse
from argparse import RawTextHelpFormatter
import sys
from os.path import expanduser

import gym
import datetime

# for graphing
from agents.Memory import Memory
from utils.summary import Summary
from training.training import train

# for parameters
import json

PATH = expanduser("~")
MODEL_FILENAME = ''


# TODO: add method comments
# TODO: add replay mode (load previous model and let it act in environments without learning)
# TODO: make epsilon decay configurable
# TODO: add DDDQN with PER as comparison option
# TODO: update gif-ing-option (store first episode, best episode from n steps)
# TODO: update model-saving-option (store first episode, best episode from n steps)

if __name__ == "__main__":

    # For more on how argparse works see documentation
    # create argument options
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-file", "--filename", help='name of file containing parameters in json-format',)

    # retrieve user inputted args from cmd line
    args = parser.parse_args()

    # Read JSON data into the datastore variable
    config_file_path = PATH + args.filename
    if config_file_path:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

# ============================================================================================================== #

    # Prepossessing folder
    # this takes care of the environment specifics and image processing
    if config['environment'] == 'Pong-v0':
        import utils.preprocessing.Pong_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Pong"
        print('Pong works')
    elif config['environment'] == 'SpaceInvaders-v0':
        import utils.preprocessing.SpaceInvaders_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "SpaceInvaders"
        print('SpaceInvaders works')
    elif config['environment'] == 'MsPacman-v0':
        import utils.preprocessing.MsPacman_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "MsPacman"
        print('MsPacman works')
    elif config['environment'] == 'Breakout-v0':
        import utils.preprocessing.Breakout_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Breakout"
        print('Breakout works')
    elif config['environment'] == 'CartPole-v1':
        import utils.preprocessing.Cartpole_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "CartPole"
        print('Cartpole works')
    else:
        sys.exit("Environment not found")

    # create gym env
    env = gym.make(config['environment'])
    # initialise processing class specific to environment
    processor = Preprocess.Processor()
    # state space is determined by the deque storing the frames from the env
    state_space = processor.get_state_space()

    if config['environment'] == 'CartPole-v1':
        state_space = (env.observation_space.shape[0],)

    # action space given by the environment
    action_space = env.action_space.n

    # ============================================================================================================== #

    # here we change the action space if it contains 'useless' keys or actions that do the same thing
    # if no useless keys it just returns the envs defined action space
    # This function is created in the preprocess file
    action_space = processor.new_action_space(action_space)

    # algorithm folder
    if config['algorithm'] == 'DQN':
        from agents.image_input.DQN_Brain import Learning
        PATH = PATH + '/Gym-T4-Testbed/output/DQN/'
        print('DQN works')
    elif config['algorithm'] == 'DoubleDQN':
        from agents.image_input.Double_DQN_Brain import Learning
        PATH = PATH + '/Gym-T4-Testbed/output/DoubleDQN/'
        print('Double works')
    elif config['algorithm'] == 'DuelingDQN':
        from agents.image_input.Dueling_Brain import Learning
        PATH = PATH + '/Gym-T4-Testbed/output/DuelingDQN/'
        print('Dueling works')
    else:
        sys.exit("Algorithm not found")

    learner = Learning(state_space, action_space, config)

    # ============================================================================================================== #

    # create memory
    memory = Memory(config['memory_size'], state_space)

    # ============================================================================================================== #

    # Graphing results
    now = datetime.datetime.now()
    # our graphing function
    # summary sets the ranges and targets and saves the graph
    graph = Summary(summary_types=['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_average_reward', 'sumiz_epsilon'],
                    # maximum exploitation value
                    epsilon_goal=0.99,
                    # desired name for file
                    name=MODEL_FILENAME + str(now),
                    # file path to save graph. i.e "/Desktop/Py/Scenario_Comparision/Maze/Model/"
                    save_path=PATH + '/graphs/',
                    # episode upper bound for graph
                    episode_max=config['episodes'],
                    # step upper bound for graph
                    step_max_m=processor.step_max,
                    # reward upper bound for graph
                    reward_min_m=processor.reward_min,
                    # reward lower bound for graph
                    reward_max_m=processor.reward_max)

    # ============================================================================================================== #

    # train learner and plot results
    train(env, learner, memory, graph, processor, config, PATH)
