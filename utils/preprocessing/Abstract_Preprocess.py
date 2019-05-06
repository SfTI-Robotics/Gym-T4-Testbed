from abc import ABC
from collections import deque

import numpy as np
from PIL import Image


# see https://github.com/rohitgirdhar/Deep-Q-Networks for preprocessing


class AbstractProcessor(ABC):

    # game-specific values
    step_max = None
    reward_min = None
    reward_max = None

    def __init__(self):
        # values needed for every processor
        self.deque = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
        self.resize_size = (110, 84)
        self.std_img = True
        self.new_size = 84

    def get_state_space(self):
        # this is used to return the processed shape  to the network initializer
        # print('image size: ', np.shape(self.deque))
        return np.shape(self.deque)

    def process_state_for_memory(self, state, is_new_episode):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eighth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        state_image = Image.fromarray(state, 'RGB')
        state_image = state_image.convert('L')  # to gray
        state_image = state_image.resize((self.new_size, self.new_size), Image.ANTIALIAS)
        state_image = np.array(state_image).astype('uint8')
        if is_new_episode:
            self.deque.append(state_image)
            self.deque.append(state_image)
            self.deque.append(state_image)
            self.deque.append(state_image)
        else:
            # append new frame to deque
            # in other steps one slot is already filled and the second one needs to be filled
            self.deque.append(state_image)
        stacked_state = np.stack(self.deque, axis=0)
        return stacked_state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        state = state.astype('float')
        # not done in nature paper, so not doing this
        if self.std_img:
            state -= 128.0
            state /= 255.0
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return [self.process_state_for_network(sample) for
                sample in samples]

    @staticmethod
    def process_reward(reward):
        """Clip reward between -1 and 1."""
        return np.sign(reward)

