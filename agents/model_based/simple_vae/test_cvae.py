from simple_vae import CVAE
import cv2
import sys
import numpy as np
import os
import imageio


env_name = sys.argv[1]

<<<<<<< HEAD
IMAGE_FOLDER_TRIPLE = './images/triple/'
IMAGE_FOLDER_CURR_PRED = './images/current_vs_predicted/'
IMAGE_FOLDER_PRED_GROUND = './images/predicted_vs_ground/'

DATA_FILE = './data/rollout/rollout-1.npz'
=======
data_file = './data/' + 'rollout_' + env_name + '/rollout-1.npz'
>>>>>>> 408a0f0... Changed model to multiplicative

frame = 201


obs_data = np.load(DATA_FILE)['obs']
action_data = np.load(DATA_FILE)['actions']
next_data = np.load(DATA_FILE)['next_frame']

for i in range(300):
    obs = obs_data[i]
    action = action_data[i]
    ground_truth = next_data[i]*255.

    current_frame = obs[:, :, 0]*255
    current_frame = np.expand_dims(current_frame, axis=3)

    obs = np.expand_dims(obs, axis=0)
    action = np.expand_dims(action, axis=0)

<<<<<<< HEAD
    predicted_next = cvae.predict(obs, action)
=======
cvae = CVAE(len(action))
cvae.set_weights('./cvae_weights_mult_' + env_name + '.h5')
>>>>>>> 408a0f0... Changed model to multiplicative

    # Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
    predicted_image = predicted_next[0, :, :, :]*255.

    triple = np.concatenate(
        (current_frame, predicted_image, ground_truth), axis=1)

    orig_vs_pred = np.concatenate((current_frame, predicted_image), axis=1)
    pred_vs_ground = np.concatenate((predicted_image, ground_truth), axis=1)

<<<<<<< HEAD
    cv2.imwrite(IMAGE_FOLDER_TRIPLE + '%03d.png' % i, triple)
    cv2.imwrite(IMAGE_FOLDER_CURR_PRED + '%03d.png' % i, orig_vs_pred)
    cv2.imwrite(IMAGE_FOLDER_PRED_GROUND + '%03d.png' % i, pred_vs_ground)
=======
cv2.imwrite('./predicted_' + env_name + '.jpg', image_vector)
cv2.imwrite('./actual_' + env_name + '.jpg', ground_truth)
>>>>>>> 408a0f0... Changed model to multiplicative
