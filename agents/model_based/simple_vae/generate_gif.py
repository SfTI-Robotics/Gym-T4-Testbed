import imageio
import os
import glob

def create_gif():
    images = []

    for infile in sorted(glob.glob('./images/*.png')):
        image = imageio.imread(infile)
        images.append(image)

<<<<<<< HEAD
imageio.mimsave('./gifs/triple_comparison.gif',
                images, duration=0.5)
=======
    imageio.mimsave('./gifs/triple_comparison.gif',
                    images, fps=15)


if __name__ == "__main__":
    create_gif()
>>>>>>> 1fd1be0... Trained models for pacman
