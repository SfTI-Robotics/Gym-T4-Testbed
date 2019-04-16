import os

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser

home = expanduser("~")

# update frequency
FREQUENCY = 1

STEP_MIN_M = -1
TIME_MIN_M = 0

STEP_MIN_F = 3
STEP_MAX_F = 12

TIME_MIN_F = 0
TIME_MAX_F = 0.025

REWARD_MIN_F = 0
REWARD_MAX_F = 1


# TODO: add smoothed plot for rewards
#  plots end up looking really spiky when plotting a lot of episodes
#  this can affect readability
#  in addition to the current rewards-plot, add a sliding-window one
#  for every episode, plot the average reward received in the last +/- n episodes
#  (same for time, steps)

# TODO: scale of execution time should adapt to actual values, without having to guess a maximum

# TODO: add plot of loss
#   https://github.com/UoA-RL/Gym-T4-Testbed/blob/henry_test/networks.py

class Summary:
    def __init__(
            self,
            # which summaries to display:
            # ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_average_reward', 'sumiz_epsilon']
            summary_types,
            # the optimal step count of the optimal policy
            step_goal=None,
            # the maximum reward for the optimal policy
            reward_goal=None,
            # maximum exploitation value
            epsilon_goal=None,
            # the 'focus' section graphs only between the start and end focus index. Useful for detail comparision
            start_focus=0,
            end_focus=0,
            # desired name for file
            name="default_image",
            # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
            save_path='',
            # episode lower bound for graph
            episode_min=0,
            # episode upper bound for graph
            episode_max=100,
            # step upper bound for graph
            step_max_m=500,
            # reward lower bound for graph
            reward_min_m=-100,
            # reward upper bound for graph
            reward_max_m=100
    ):
        self.summary_types = summary_types
        self.step_goal = step_goal
        self.reward_goal = reward_goal
        self.epsilon_goal = epsilon_goal
        self.start_focus = start_focus
        self.end_focus = end_focus
        self.general_filename = name
        self.save_path = save_path
        self.EPISODE_MIN = episode_min
        self.EPISODE_MAX = episode_max
        self.STEP_MAX_M = step_max_m
        self.REWARD_MIN_M = reward_min_m
        self.REWARD_MAX_M = reward_max_m

        self.step_summary = []
        self.time_summary = []
        self.reward_summary = []
        self.epsilon_summary = []
        self.average_reward_summary = []

        # initialize the number of main axis
        self.num_main_axes = 0
        self.is_average_reward_axes = False

        # determines number of graph we want to plot in the figure
        if 'sumiz_step' in self.summary_types:
            self.num_main_axes += 1
        if 'sumiz_time' in self.summary_types:
            self.num_main_axes += 1
        if 'sumiz_reward' in self.summary_types:
            self.num_main_axes += 1
        if 'sumiz_epsilon' in self.summary_types:
            self.num_main_axes += 1
        if 'sumiz_average_reward' in self.summary_types:
            self.num_main_axes += 1
            self.is_average_reward_axes = True

        # create folder, if necessary
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.num_focus_axes = 0
        if 'sumiz_step' in self.summary_types:
            self.num_focus_axes += 1
        if 'sumiz_time' in self.summary_types:
            self.num_focus_axes += 1
        if 'sumiz_epsilon' in self.summary_types:
            self.num_focus_axes += 1

        if self.step_goal is not None:
            self.average_reward_goal = self.reward_goal / float(self.step_goal)
        else:
            self.average_reward_goal = None

    def summarize(
            self,
            # for the current iteration
            episode_count,
            # an array that records steps taken in each episode. Index indicates episode
            step_count=1,
            # an array that records the operation time for each episode
            time_count=0,
            # an array that records total reward collected in each episode
            reward_count=0,
            # epsilon greedy value
            epsilon_value=0,
            e_greedy_formula='e-greedy formula = '
    ):
        self.update(step_count, time_count, reward_count, epsilon_value)

        # generate summary graphs
        if episode_count % FREQUENCY == 0:
            self.plot_summary_graphs(e_greedy_formula)

        # generate index-focused summary graph
        if self.num_focus_axes != 0 and self.start_focus != self.end_focus:
            if episode_count % FREQUENCY == 0 and self.start_focus < episode_count < self.end_focus:
                self.plot_index_focused_summary_graphs(episode_count)

    def plot_summary_graphs(self, e_greedy_formula):
        fig1 = plt.figure(figsize=(5, 10))  # plotting normally takes ~0.5 seconds
        i = 1
        if 'sumiz_step' in self.summary_types:
            ax1 = fig1.add_subplot(self.num_main_axes, 1, i)
            # plt.axis([self.EPISODE_MIN, self.EPISODE_MAX, STEP_MIN_M, self.STEP_MAX_M])
            plt.axis([self.EPISODE_MIN, self.EPISODE_MAX, 0, np.max(self.step_summary)])
            ax1.plot(range(len(self.step_summary)), self.step_summary)
            # only plot additional line if goal was specified
            if self.step_goal is not None:
                ax1.plot(range(len(self.step_summary)), np.repeat(self.step_goal, len(self.step_summary)), 'r:')
            ax1.set_title('Number of steps taken in each episode')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Steps taken')
            i += 1
        if 'sumiz_time' in self.summary_types:
            ax2 = fig1.add_subplot(self.num_main_axes, 1, i)
            plt.axis([self.EPISODE_MIN, self.EPISODE_MAX, 0, np.max(self.time_summary)])
            ax2.plot(range(len(self.time_summary)), self.time_summary)
            ax2.set_title('Execution time in each episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Execution time (s)')
            i += 1
        if 'sumiz_reward' in self.summary_types:
            ax3 = fig1.add_subplot(self.num_main_axes, 1, i)
            # plt.axis([self.EPISODE_MIN, self.EPISODE_MAX, self.REWARD_MIN_M, self.REWARD_MAX_M])
            plt.axis([self.EPISODE_MIN, self.EPISODE_MAX,
                      np.min([0, np.min(self.reward_summary)]), np.max(self.reward_summary)])
            ax3.plot(range(len(self.reward_summary)), self.reward_summary)
            # only plot additional line if goal was specified
            if self.reward_goal is not None:
                ax3.plot(range(len(self.reward_summary)), np.repeat(self.reward_goal, len(self.reward_summary)),
                         'r:')
            ax3.set_title('Reward in each episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            i += 1
        if 'sumiz_epsilon' in self.summary_types:
            ax4 = fig1.add_subplot(self.num_main_axes, 1, i)
            plt.axis([self.EPISODE_MIN, self.EPISODE_MAX, 0, 1])
            ax4.plot(range(len(self.epsilon_summary)), self.epsilon_summary, label=e_greedy_formula)
            # only plot additional line if goal was specified
            if self.epsilon_goal is not None:
                ax4.plot(range(len(self.epsilon_summary)), np.repeat(self.epsilon_goal, len(self.epsilon_summary)),
                         'r:')
            ax4.set_title('Epsilon Greedy')
            ax4.set_xlabel('Episode \n ' + e_greedy_formula)
            ax4.set_ylabel('Epsilon')
            i += 1
        if 'sumiz_average_reward' in self.summary_types:
            ax5 = fig1.add_subplot(self.num_main_axes, 1, i)
            ax5.plot(range(len(self.average_reward_summary)), self.average_reward_summary)
            # only plot additional line if goal was specified
            if self.average_reward_goal is not None:
                ax5.plot(range(len(self.average_reward_summary)),
                         np.repeat(self.average_reward_goal, len(self.average_reward_summary)), 'r:')
            ax5.set_title('Reward in each episode per step')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Reward per step')
            i += 1
        plt.tight_layout()
        fig1.savefig(self.save_path + self.general_filename + ".png")
        plt.close(fig1)

    def plot_index_focused_summary_graphs(self, episode_count):
        fig2 = plt.figure(figsize=(5, 5))
        i = 1
        if 'sumiz_step' in self.summary_types:
            ax1 = fig2.add_subplot(self.num_focus_axes, 1, i)
            plt.axis([self.start_focus, self.end_focus, STEP_MIN_F, STEP_MAX_F])
            ax1.plot(range(self.start_focus, min(episode_count, self.end_focus)),
                     self.step_summary[self.start_focus:min(episode_count, self.end_focus)])
            # only plot additional line if goal was specified
            if self.step_goal is not None:
                ax1.plot(range(self.start_focus, min(episode_count, self.end_focus)),
                         np.repeat(self.step_goal, min(episode_count, self.end_focus) - self.start_focus), 'r:')
            ax1.set_title('Number of steps taken in each episode')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Steps taken')
            i += 1
        if 'sumiz_time' in self.summary_types:
            ax2 = fig2.add_subplot(self.num_focus_axes, 1, i)
            plt.axis([self.start_focus, self.end_focus, TIME_MIN_F, TIME_MAX_F])
            ax2.plot(range(self.start_focus, min(episode_count, self.end_focus)),
                     self.time_summary[self.start_focus:min(episode_count, self.end_focus)])
            ax2.set_title('Execution time in each episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Execution time (s)')
            i += 1
        if 'sumiz_epsilon' in self.summary_types:
            ax3 = fig2.add_subplot(self.num_focus_axes, 1, i)
            ax3.plot(range(self.start_focus, min(episode_count, self.end_focus)),
                     self.epsilon_summary[self.start_focus:min(episode_count, self.end_focus)])
            # only plot additional line if goal was specified
            if self.epsilon_goal is not None:
                ax3.plot(range(self.start_focus, min(episode_count, self.end_focus)),
                         np.repeat(self.epsilon_goal, min(episode_count, self.end_focus) - self.start_focus), 'r:')
            ax3.set_title('Epsilon Greedy')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            i += 1
        plt.tight_layout()
        fig2.savefig(self.save_path + self.general_filename + "_focused_summary.png")
        plt.close(fig2)

    def update(self, step_count, time_count, reward_count, epsilon_value):
        self.step_summary.append(step_count)
        self.time_summary.append(time_count)
        self.reward_summary.append(reward_count)
        self.epsilon_summary.append(epsilon_value)

        if self.is_average_reward_axes:
            # check for divide by zero error
            if step_count == 0:
                print("Step array contains zero(s). Reward-per-step graph will be omitted.")
                self.average_reward_summary.append(0)
            else:
                # find average reward in each episode
                self.average_reward_summary.append(reward_count / float(step_count))

    @staticmethod
    def display_parameters(initial_epsilon=None, max_epsilon=None, learning_rate=None, reward_decay=None,
                           memory_size=None):
        print('=' * 40)
        print('{}{}{}'.format('=' * 11, ' Hyper-parameters ', '=' * 11))
        print('=' * 40)
        print('{}{}'.format(' Starting Epsilon: ', initial_epsilon))
        print('{}{}'.format(' Maximum Epsilon: ', max_epsilon))
        print('{}{}'.format(' Learning Rate (Alpha): ', learning_rate))
        print('{}{}'.format(' Reward Decay (Gamma): ', reward_decay))
        print('{}{}'.format(' Memory Size: ', memory_size))
        print('=' * 40)
