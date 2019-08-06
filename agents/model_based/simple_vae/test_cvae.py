from simple_vae import CVAE
import cv2
import numpy as np
import os
import imageio


cvae = CVAE()
cvae.set_weights('./cvae_weights.h5')

IMAGE_FOLDER = './images/'
DATA_FILE = './data/rollout/rollout-1.npz'

frame = 200


obs_data = np.load(DATA_FILE)['obs']
action_data = np.load(DATA_FILE)['actions']
next_data = np.load(DATA_FILE)['next_frame']

for i in range(300):
    obs = obs_data[i]
    action = action_data[i]
    ground_truth = next_data[i]*255.

    obs = np.expand_dims(obs,axis = 0)
    action = np.expand_dims(action,axis=0)

    predicted_next = cvae.predict(obs, action)

    # Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
    predicted_image = predicted_next[0,:,:,:]*255.

    side_by_side = np.concatenate((predicted_image, ground_truth), axis=1)
    cv2.imwrite(IMAGE_FOLDER + '%03d.png' %i, side_by_side)

