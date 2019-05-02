import argparse
import datetime
import json
import time
from os.path import expanduser

import gym
from numpy.random import seed
from tensorflow import set_random_seed

import utils.preprocessing.Cartpole_Preprocess as Preprocess
from agents.image_input.Hybrid_Brain import Learning
from agents.memory import Memory
from training.testing_functions import test
from utils.summary import Summary

seed(2)
set_random_seed(2)

PATH = expanduser("~")
MODEL_FILENAME = ''


def test_hybrid():
    print('# =========================================== TEST DQN =========================================== #')
    prev_epsilon = learner.dqn_agent.epsilon
    prev_min_epsilon = learner.dqn_agent.min_epsilon
    prev_switch = learner.switch

    # don't act randomly
    learner.switch = True
    learner.dqn_agent.epsilon = 0
    learner.dqn_agent.min_epsilon = 0

    test(learner.dqn_agent, env, MODEL_FILENAME, PATH + '/test_dqn/', config, processor, min_reward=0, max_reward=500)

    # ============================================================================================================== #

    print('# =========================================== TEST PG =========================================== #')

    learner.switch = False

    test(learner.pg_agent, env, MODEL_FILENAME, PATH + '/test_pg/', config, processor, min_reward=0, max_reward=500)

    learner.dqn_agent.epsilon = prev_epsilon
    learner.dqn_agent.min_epsilon = prev_min_epsilon
    learner.switch = prev_switch


if __name__ == "__main__":

    # For more on how argparse works see documentation
    # create argument options
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-file", "--filename", help='name of file containing parameters in json-format', )

    # retrieve user inputted args from cmd line
    args = parser.parse_args()

    # Read JSON data into the datastore variable
    config_file_path = PATH + args.filename
    if config_file_path:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

    # ============================================================================================================== #

    MODEL_FILENAME = MODEL_FILENAME + "CartPole"

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

    # here we change the action space if it contains 'useless' keys or actions that do the same thing
    # if no useless keys it just returns the envs defined action space
    # This function is created in the preprocess file
    action_space = processor.new_action_space(action_space)

    # ============================================================================================================== #

    PATH = PATH + '/Gym-T4-Testbed/hybrid/output/Hybrid/'
    learner = Learning(state_space, action_space, config)

    # ============================================================================================================== #

    # create memory
    dqn_memory = Memory.Memory(config['memory_size'], state_space)
    pg_memory = Memory.Memory(config['memory_size'], state_space)

    # ============================================================================================================== #

    # train learner and plot results
    # summary_writer = tensorflow.summary.FileWriter(PATH + 'tensorboard_summary/')
    summary = Summary(['sumiz_step', 'sumiz_reward', 'sumiz_epsilon'],
                      name=MODEL_FILENAME + str(datetime.datetime.now()),
                      save_path=PATH + '/graphs/',
                      min_reward=0,
                      max_reward=500)
    summary_rewards, summary_epsilons, summary_steps = [], [], []

    # ============================================================================================================== #

    training_step = 0

    # for episode in range(int(episodes)):
    for episode in range(config['episodes']):

        state = env.reset()

        # Processing initial image cropping, grayscale, and stacking 4 of them
        state = processor.preprocessing(state, True)

        start_time = time.time()
        sum_rewards_array = 0  # total rewards for graphing
        step = 0  # count total steps for each episode for the graph

        while True:
            action = learner.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if config['environment'] == 'CartPole-v1':
                # punish if terminal state reached
                if done:
                    reward = -reward

            sum_rewards_array += reward
            next_state = processor.preprocessing(next_state, False)

            dqn_memory.store_transition(state, action, reward, next_state, done)
            pg_memory.store_transition(state, action, reward, next_state, done)

            step += 1
            state = next_state
            training_step += 1

            # train dqn with batch data at every step
            if len(dqn_memory.stored_transitions) > config['initial_exploration_steps']:
                states, actions, rewards, next_states, dones = dqn_memory.sample(config['batch_size'])
                learner.dqn_agent.train_hybrid_network(states, actions, rewards, next_states, dones, training_step,
                                                       learner.switch)

            if done:
                if episode == config['switch_steps']:
                    test_hybrid()
                    print(
                        '# =========================================== SWITCH =========================================== #')
                    learner.switch = True
                    # learner.update_weights()

                # train pg with episode data after every episode
                states, actions, rewards, next_states, dones = pg_memory.sample_all()
                learner.pg_agent.train_network(states, actions, rewards, next_states, dones, step)
                # learner.dqn_agent.train_hybrid_network(states, actions, rewards, next_states, dones, step, learner.switch)

                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.dqn_agent.epsilon,
                      ', steps = ', step,
                      ", total reward = ", sum_rewards_array)
                summary_rewards.append(sum_rewards_array)
                summary_steps.append(step)
                summary_epsilons.append(learner.dqn_agent.epsilon)

                if episode % 50 == 0:
                    summary.summarize(step_counts=summary_steps, reward_counts=summary_rewards,
                                      epsilon_values=summary_epsilons,
                                      e_greedy_formula=learner.dqn_agent.e_greedy_formula)
                    summary_rewards, summary_epsilons, summary_steps = [], [], []
                    test_hybrid()

                break

        # ============================================================================================================== #

        # if config['save_tensorboard_summary']:
            # save episode data to tensorboard summary
            # save_episode_to_summary(summary_writer, episode, step, time.time() - start_time,
            #                         sum_rewards_array, learner.epsilon)

    # ============================================================================================================== #

    summary.summarize(step_counts=summary_steps, reward_counts=summary_rewards, epsilon_values=summary_epsilons,
                      e_greedy_formula=learner.dqn_agent.e_greedy_formula)

    # ============================================================================================================== #

    test_hybrid()

    # ============================================================================================================== #

    # killing environment to prevent memory leaks
    env.close()
