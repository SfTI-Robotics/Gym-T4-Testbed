import imageio
import os
import glob

images = []

for infile in sorted(glob.glob('./images/*.png')):
    image = imageio.imread(infile)
    images.append(image)

imageio.mimsave('./gifs/triple_comparison.gif',
                images, duration=0.5)
