import argparse
import copy
import datetime
import json
import time
from os.path import expanduser

import gym
import tensorflow
from numpy.random import seed
from tensorflow import set_random_seed

import utils.preprocessing.Cartpole_Preprocess as Preprocess
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

summary_frequency = 50000
test_frequency = 50000


def test_hybrid():
    print('# =========================================== TEST DQN =========================================== #')

    test(learner.dqn_agent, copy.deepcopy(env), config, copy.deepcopy(processor),
         'test_' + config['environment'] + '_' + str(datetime.datetime.now()), PATH + '/test_dqn/')

    # ============================================================================================================== #

    print('# =========================================== TEST PG =========================================== #')

    test(learner.pg_agent, copy.deepcopy(env), config, copy.deepcopy(processor),
         'test_' + config['environment'] + '_' + str(datetime.datetime.now()), PATH + '/test_pg/')


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

    MODEL_FILENAME = MODEL_FILENAME + "CartPole"
    env = gym.make(config['environment'])
    processor = Preprocess.Processor()
    state_space = processor.get_state_space()
    if config['environment'] == 'CartPole-v1':
        state_space = (env.observation_space.shape[0],)
    action_space = env.action_space.n

    # ============================================================================================================== #

    PATH = PATH + '/Gym-T4-Testbed/output/Hybrid/'
    learner = Learning(state_space, action_space, config)

    # ============================================================================================================== #

    # loading neural network weights and parameters
    if config['load_model']:
        learner.load_network(PATH + 'models/', config['model_file'])

    # ============================================================================================================== #

    dqn_memory = Memory.Memory(config['memory_size'], state_space)
    pg_memory = Memory.Memory(config['memory_size'], state_space)

    # ============================================================================================================== #

    # train learner and plot results
    summary_writer = tensorflow.summary.FileWriter(PATH + 'tensorboard_summary/')
    summary = Summary(['sumiz_step', 'sumiz_reward', 'sumiz_epsilon'],
                      name=MODEL_FILENAME + str(datetime.datetime.now()),
                      save_path=PATH + '/graphs/',
                      min_reward=0,
                      max_reward=500)
    summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

    state = env.reset()
    state = processor.process_state_for_memory(state, True)
    episode_start_time = time.time()
    episode_reward = 0
    episode_steps = 0
    episode = 0

    # ============================================================================================================== #

    for step in range(config['steps']):

        # perform next step
        action = learner.choose_action(processor.process_state_for_network(state))
        next_state, reward, done, _ = env.step(action)
        reward = processor.process_reward(reward)
        next_state = processor.process_state_for_memory(next_state, False)

    # ============================================================================================================== #

        if config['environment'] == 'CartPole-v1':
            # punish if terminal state reached
            if done:
                reward = -reward

        # update memories
        dqn_memory.store_transition(state, action, reward, next_state, done)
        pg_memory.store_transition(state, action, reward, next_state, done)

        # update counters
        state = next_state
        episode_reward += reward
        episode_steps += 1

        # train dqn with batch data at every step
        # if len(dqn_memory.stored_transitions) > config['initial_exploration_steps']:
        if len(dqn_memory.stored_transitions) > config['initial_exploration_steps'] and learner.switch:
            states, actions, rewards, next_states, dones = dqn_memory.sample(config['batch_size'], processor)
            learner.train_dqn_network(states, actions, rewards, next_states, dones, step)

    # ============================================================================================================== #

        if done:

            # print episode results
            print('Completed Episode = ' + str(episode),
                  ', epsilon =', "%.4f" % learner.dqn_agent.epsilon,
                  ", reward = ", episode_reward,
                  ", steps = ", episode_steps,
                  ', total steps = ', step,
                  ", episode time = ", "{0:.2f}".format(time.time() - episode_start_time))

            # train pg with episode data after every episode
            states, actions, rewards, next_states, dones = pg_memory.sample_all(processor)
            learner.train_pg_network(states, actions, rewards, next_states, dones, step)
            # learner.train_dqn_network(states, actions, rewards, next_states, dones, training_step)

            # update plot summary
            summary_rewards.append(episode_reward)
            summary_steps.append(episode_steps)
            summary_epsilons.append(learner.dqn_agent.epsilon)

            # save episode data to tensorboard summary
            if config['save_tensorboard_summary']:
                save_episode_to_summary(summary_writer, episode, step, time.time() - episode_start_time,
                                        episode_reward, learner.epsilon)

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
        if (step % summary_frequency == 0 and step != 0) or step == config['steps'] - 1:
            summary.summarize(step_counts=summary_steps, reward_counts=summary_rewards,
                              epsilon_values=summary_epsilons,
                              e_greedy_formula=learner.dqn_agent.e_greedy_formula)
            summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

        if step % test_frequency == 0:
            test_hybrid()

    # ============================================================================================================== #

        if config['save_model'] and \
                ((step != 0 and step % config['model_save_frequency'] == 0) or step == config['steps'] - 1):
            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            learner.save_network(PATH + 'models/', config['environment'] + '_' + str(episode))

    test_hybrid()
    env.close()
