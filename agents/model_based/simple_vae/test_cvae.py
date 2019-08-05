from simple_vae import CVAE
import cv2
import numpy as np

cvae = CVAE()
cvae.print_layer_shapes()
cvae.set_weights('./cvae_weights.h5')

data_file = './data/rollout/rollout-1.npz'

obs_data = np.load(data_file)['obs']
obs = obs_data[0]

action_data = np.load(data_file)['actions']
action = action_data[0]

next_data = np.load(data_file)['next_frame']
ground_truth = next_data[0]

print(obs.shape)
print(action.shape)
print(ground_truth.shape)

obs = np.expand_dims(obs,axis = 0)
action = np.expand_dims(action,axis=0)


predicted_image = cvae.predict(obs,action)
# Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
image_vector = predicted_image[0,:,:,:]*255

cv2.imwrite('./predicted.jpg', image_vector)
