# from vae import VAE
import gym
import os
import cv2
from vae import VAE
import numpy as np

image = cv2.imread('./resized_img.jpg')
print(image.shape)

vae = VAE()
vae.set_weights('./vae_weights.h5')

predicted_image = vae.predict([image])

cv2.imwrite('./predicted.jpg', predicted_image)

