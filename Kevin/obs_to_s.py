import numpy as np



def pong_preprocess(image):
    image = image[35:195]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1
    
    return image.astype(np.float).ravel()