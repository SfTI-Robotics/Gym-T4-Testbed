import argparse
import copy
import datetime
import json
import sys
import time
from os.path import expanduser

import gym
import tensorflow
from numpy.random import seed
from tensorflow import set_random_seed

from agents.hybrids.pg_dqn_Brain import Learning
from agents.memory import Memory
from training.testing_functions import test
from utils.storing import save_episode_to_summary
from utils.summary import Summary

seed(2)
set_random_seed(2)

HOME_PATH = expanduser("~")
PATH = expanduser("~")
MODEL_FILENAME = ''


def test_hybrid():
    print('# =========================================== TEST DQN =========================================== #')
    test(learner.dqn_agent.get_test_learner(), copy.deepcopy(env), config, copy.deepcopy(processor),
         config['environment'] + '_' + str(datetime.datetime.now()),
         PATH + '/test_dqn/', episode=episode)

    # ============================================================================================================== #

    print('# =========================================== TEST PG =========================================== #')
    test(learner.ppo_agent.get_test_learner(), copy.deepcopy(env), config, copy.deepcopy(processor),
         config['environment'] + '_' + str(datetime.datetime.now()),
         PATH + '/test_pg/', episode=episode)


if __name__ == "__main__":
    # combination of run_main and train, adapted for pg+dqn hybrid

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-file", "--filename", help='name of file containing parameters in json-format', )
    args = parser.parse_args()

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
    elif config['environment'] == 'Enduro-v0':
        import utils.preprocessing.Enduro_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "Enduro"
        print('Breakout works')
    elif config['environment'] == 'CartPole-v1':
        import utils.preprocessing.Cartpole_Preprocess as Preprocess
        MODEL_FILENAME = MODEL_FILENAME + "CartPole"
        print('Cartpole works')
    else:
        sys.exit("Environment not found")

    # create gym env
    env = gym.make(config['environment'])
    # add seed to make results reproducible
    env.seed(0)

    # initialise processing class specific to environment
    processor = Preprocess.Processor()
    # state space is determined by the deque storing the frames from the env
    state_space = processor.get_state_space()

    if config['environment'] == 'CartPole-v1':
        state_space = (env.observation_space.shape[0],)

    # action space given by the environment
    action_space = env.action_space.n

    # ============================================================================================================== #

    PATH = PATH + '/Gym-T4-Testbed/output/Hybrid/'
    learner = Learning(state_space, action_space, config)

    # ============================================================================================================== #

    # loading neural network weights and parameters
    if config['load_model']:
        learner.load_network(PATH + 'models/', config['model_file'])

    # ============================================================================================================== #

    dqn_memory = Memory.RandomBatchMemory(config['memory_size'], state_space)
    ppo_memory = Memory.EpisodicMemory(config['memory_size'], state_space, action_space)

    # ============================================================================================================== #

    # train learner and plot results
    summary_writer = tensorflow.summary.FileWriter(PATH + 'tensorboard_summary/')
    summary = Summary(['sumiz_step', 'sumiz_reward', 'sumiz_epsilon'],
                      name=MODEL_FILENAME + str(datetime.datetime.now()),
                      save_path=PATH + '/graphs/',
                      min_reward=processor.reward_min,
                      max_reward=processor.reward_max)
    summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

    state = env.reset()
    state = processor.process_state_for_memory(state, True)
    episode_start_time = time.time()
    episode_reward = 0
    episode_steps = 0
    episode = 0
    switch_counter = 0

    # ============================================================================================================== #

    for step in range(config['steps']):

        # perform next step
        action, policy = learner.choose_action(processor.process_state_for_network(state))
        next_state, reward, done, _ = env.step(action)
        reward = processor.process_reward(reward, reward_clipping=config['reward_clipping'])
        next_state = processor.process_state_for_memory(next_state, False)

    # ============================================================================================================== #

        if config['environment'] == 'CartPole-v1':
            # punish if terminal state reached
            if done:
                reward = -reward

        # update memories
        dqn_memory.add_tuple(state, action, reward, next_state, done)
        ppo_memory.add_tuple(state, action, reward, next_state, done, policy=policy)

        # update counters
        state = next_state
        episode_reward += reward
        episode_steps += 1

        # train dqn with batch data at every step
        # if len(dqn_memory.tuples) > config['initial_exploration_steps'] and learner.switch:
        if len(dqn_memory.tuples) > config['initial_exploration_steps']:
            states, actions, rewards, next_states, dones = dqn_memory.sample(processor, batch_size=config['batch_size'])
            learner.train_dqn_network(states, actions, rewards, next_states, dones, step)

        # train ppo with batches of collected memory
        if ppo_memory.get_memory_size() > config['horizon']:
            states, actions, rewards, next_states, dones, policies = ppo_memory.sample(processor)
            learner.train_pg_network(states, actions, rewards, next_states, dones, policies)

    # ============================================================================================================== #

        if done:
            # print episode results
            print('Completed Episode = ' + str(episode),
                  ', epsilon =', "%.4f" % learner.dqn_agent.epsilon,
                  ", reward = ", episode_reward,
                  ", steps = ", episode_steps,
                  ', total steps = ', step,
                  ", episode time = ", "{0:.2f}".format(time.time() - episode_start_time))
            # update plot summary
            summary_rewards.append(episode_reward)
            summary_steps.append(episode_steps)
            summary_epsilons.append(learner.dqn_agent.epsilon)

            # save episode data to tensorboard summary
            if config['save_tensorboard_summary']:
                save_episode_to_summary(summary_writer, episode, step, time.time() - episode_start_time,
                                        episode_reward, learner.dqn_agent.epsilon)

    # ============================================================================================================== #

            # reset episode data
            episode += 1
            state = env.reset()
            episode_frames = [state]
            state = processor.process_state_for_memory(state, True)
            episode_start_time = time.time()
            episode_reward = 0
            episode_steps = 0

    # ============================================================================================================== #

        # update plot summary
        if (step % config['save_plot_frequency'] == 0 and step != 0) or step == config['steps'] - 1:
            summary.summarize(step_counts=summary_steps, reward_counts=summary_rewards,
                              epsilon_values=summary_epsilons,
                              e_greedy_formula=learner.dqn_agent.e_greedy_formula)
            summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

        if step % config['test_frequency'] == 0:
            test_hybrid()

    # ============================================================================================================== #

        if config['save_model'] and \
                ((step != 0 and step % config['save_model_frequency'] == 0) or step == config['steps'] - 1):
            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            learner.save_network(PATH + 'models/', config['environment'] + '_' + str(episode))

    test_hybrid()
    env.close()
