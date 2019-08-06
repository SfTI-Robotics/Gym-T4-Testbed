import imageio
import os
import glob

with imageio.get_writer('./images/comparison.gif', mode='I') as writer:
    for infile in sorted(glob.glob('./images/*.png')):
        image = imageio.imread(infile)
        writer.append_data(image)