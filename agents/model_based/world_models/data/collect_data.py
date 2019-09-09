import os
import random

import numpy as np
import gym
from scipy.misc import imresize

# in rollouts
NUM_EPISODES = 2000
EPISODE_LEN = 300
FRAME_SKIP = 5
INIT_STEPS = 60
CHECKPOINT_INTERVAL = 100
DATA_DIR = './data'


    