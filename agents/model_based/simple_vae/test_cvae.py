from simple_vae import CVAE
import cv2
import sys
import numpy as np
import os
import imageio


env_name = sys.argv[1]

data_file = './data/world_models/' + 'rollout_' + env_name + '/rollout-1.npz'

frame =199


obs_data = np.load(DATA_FILE)['obs']
action_data = np.load(DATA_FILE)['actions']
next_data = np.load(DATA_FILE)['next_frame']

cvae = CVAE(len(action))
cvae.set_weights('./cvae_weights_mult_' + env_name + '.h5')

for i in range(300):
    obs = obs_data[i]
    action = action_data[i]
    ground_truth = next_data[i]*255.

    current_frame = obs[:, :, 0]*255
    current_frame = np.expand_dims(current_frame, axis=3)

    obs = np.expand_dims(obs, axis=0)
    action = np.expand_dims(action, axis=0)

    predicted_next = cvae.predict(obs, action)


    # Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
    predicted_image = predicted_next[0, :, :, :]*255.

    triple = np.concatenate(
        (current_frame, predicted_image, ground_truth), axis=1)

predicted_image = cvae.predict(obs)
# Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
image_vector = predicted_image[0,:,:,:]*255.

cv2.imwrite('./predicted_' + env_name + '.jpg', image_vector)
cv2.imwrite('./actual_' + env_name + '.jpg', ground_truth)

