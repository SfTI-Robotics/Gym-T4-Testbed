import os
from os.path import expanduser

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


HOME_PATH = expanduser("~")


# Some global variables to define the whole run
test_episodes = 10
GAMMA = 0.99

bars = None
heights = None
values = None

colors = ['blue', 'red', 'green', 'yellow', 'brown', 'orange', 'pink', 'purple', 'grey']


def init():
    return bars


def animate_bars(i):
    global bars, heights, values

    heights = [values[i, val_pos] for val_pos in range(values.shape[1])]

    for h, b in zip(heights, bars):
        b.set_height(h)

    return bars


def animate_bar(i):
    global bars, heights

    bars.set_height(heights[i])

    return bars


def animate_line(frame, line):
    """
    Animation function. Takes the current frame number (to select the potion of
    data to plot) and a line object to update.
    """

    # Not strictly necessary, just so we know we are stealing these from
    # the global scope
    global values, heights

    # We want up-to and _including_ the frame'th element
    current_x_data = values[: frame + 1]
    current_y_data = heights[: frame + 1]

    line.set_xdata(current_x_data)
    line.set_ydata(current_y_data)

    # This comma is necessary!
    return line,


def load_stuff_from_file(file_path, train_episode, test_episode):
    base_name = 'test' + str(test_episode)
    base_name = 'episode' + str(train_episode) + '_' + base_name
    # path += 'episode' + str(train_episode) + '/'
    # create folder, if necessary
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    rewards = np.loadtxt(file_path + base_name + '_rewards.txt')
    actions = np.loadtxt(file_path + base_name + '_actions.txt')
    predictions = np.loadtxt(file_path + base_name + '_predictions.txt')
    return rewards, actions, predictions


def plot_predictions_bars(predictions, file_path, filename):
    global bars, heights, values

    values = predictions

    fig = plt.figure()

    position = np.arange(predictions.shape[1]) + .5

    plt.tick_params(axis='x')
    plt.tick_params(axis='y')

    heights = np.zeros(predictions.shape[1])
    rectangles = plt.bar(position, heights, align='center', color='#b8ff5c')

    bars = [element for element in rectangles]
    for pos in range(len(bars)):
        bars[pos].set_facecolor(colors[pos % len(colors)])

    plt.xticks(position, np.arange(predictions.shape[0]))

    plt.xlabel('Actions')
    plt.ylabel('Predicted values')
    plt.title('Predicted values per state')

    plt.ylim((np.min(predictions) - 1, np.max(predictions) + 1))
    plt.xlim((0, predictions.shape[1]))

    plt.grid(True)

    animation = FuncAnimation(fig, animate_bars, init_func=init, frames=predictions.shape[0], interval=1000 / 25)
    animation.save(file_path + filename + "_predictions.mp4")
    plt.close(fig)


def plot_discounted_reward_bar(rewards, file_path, filename):
    global bars, heights

    heights = discount_rewards(rewards)

    fig = plt.figure()

    plt.tick_params(axis='x')
    plt.tick_params(axis='y')

    rectangles = plt.bar([.5], [0], align='center', color='#b8ff5c')

    bars = rectangles[0]
    bars.set_facecolor(colors[0])

    plt.xlabel('Actions')
    plt.ylabel('Predicted Q-values')
    plt.title('Predicted Q-values per state')

    plt.ylim((np.min(heights) - 1, np.max(heights) + 1))
    plt.xlim((0, 1))

    plt.grid(False)

    animation = FuncAnimation(fig, animate_bar, init_func=init, frames=len(heights), interval=1000 / 25)
    animation.save(file_path + filename + "_discounted_reward_bar.mp4")
    plt.close(fig)


def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * GAMMA + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def plot_discounted_reward_lines(rewards, file_path, filename):
    global heights, values

    heights = discount_rewards(rewards)
    values = np.arange(len(heights))

    # Now we can do the plotting!
    fig, ax = plt.subplots(1)
    # Initialise our line
    line, = ax.plot([0], [0])

    # Have to set these otherwise we will get one ugly plot!
    ax.set_xlim(0, len(heights))
    ax.set_ylim(np.min(heights) - 1, np.max(heights) + 1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Discounted reward from here on")

    # Make me pretty
    fig.tight_layout()

    animation = FuncAnimation(
        # Your Matplotlib Figure object
        fig,
        # The function that does the updating of the Figure
        animate_line,
        # Frame information (here just frame number)
        len(heights),
        # Extra arguments to the animate function
        fargs=[line],
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval=1000 / 25
    )
    animation.save(file_path + filename + "_discounted_rewards_line.mp4")
    plt.close(fig)


def plot_all_predictions(file_path, train_episode):
    for e in range(test_episodes):
        rewards, actions, predictions = load_stuff_from_file(file_path, train_episode, e)
        plot_discounted_reward_bar(rewards, file_path, 'episode' + str(train_episode) + 'test' + str(e))
        plot_discounted_reward_lines(rewards, file_path, 'episode' + str(train_episode) + 'test' + str(e))
        plot_predictions_bars(predictions, file_path, 'episode' + str(train_episode) + 'test' + str(e))


# used to save rewards, actions, predictions to txt-files
def save_stuff_to_file(episode_rewards, episode_actions, predictions, test_episode, file_path, train_episode=None):
    base_name = 'test' + str(test_episode)
    if train_episode is not None:
        base_name = 'episode' + str(train_episode) + '_' + base_name
        file_path += 'episode' + str(train_episode) + '/'
        # create folder, if necessary
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    np.savetxt(file_path + base_name + '_rewards.txt', episode_rewards, fmt='%d')
    np.savetxt(file_path + base_name + '_actions.txt', episode_actions, fmt='%d')
    np.savetxt(file_path + base_name + '_predictions.txt', predictions)


if __name__ == "__main__":
    path = HOME_PATH + '/Gym-T4-Testbed/output/Hybrid/test_dqn/DQN/episode743/'
    plot_all_predictions(path, 743)
    path = HOME_PATH + '/Gym-T4-Testbed/output/Hybrid/test_dqn/DQN/episode743/'
    plot_all_predictions(path, 743)
