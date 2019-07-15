# from vae import VAE
import gym
import os
import cv2
from vae import VAE
import numpy as np

# Might need to change this to read in your version of 
image = cv2.imread('./original_img.jpg')
new_image = cv2.resize(image[40:200,:], (64,64), interpolation = cv2.INTER_AREA)/255
vae = VAE()
vae.print_layer_shapes()
vae.set_weights('./vae_weights.h5')

# Need to give the input an extra dimension since model.predict() expects batch size as well [batch_size, img_height, img_width, channels]
new_image = np.expand_dims(new_image,axis = 0)
predicted_image = vae.predict(new_image)

# Have to use the extra layer and multiply rgb values by 255 to get the original image since before we divided by 255 during storage
image_vector = predicted_image[0,:,:,:]*255

cv2.imwrite('./predicted.jpg', image_vector)

