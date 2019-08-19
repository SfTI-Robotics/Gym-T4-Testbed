import imageio
import os
import glob

triple_images = []
curr_vs_pred_images = []
pred_vs_ground_images = []

for infile in sorted(glob.glob('./images/triple/*.png')):
    image = imageio.imread(infile)
    triple_images.append(image)

for infile in sorted(glob.glob('./images/current_vs_predicted/*.png')):
    image = imageio.imread(infile)
    curr_vs_pred_images.append(image)

for infile in sorted(glob.glob('./images/predicted_vs_ground/*.png')):
    image = imageio.imread(infile)
    pred_vs_ground_images.append(image)

imageio.mimsave('./gifs/triple_comparison.gif',
                triple_images, duration=0.5)
imageio.mimsave('./gifs/curr_pred_comparison.gif',
                curr_vs_pred_images, duration=0.5)
imageio.mimsave('./gifs/pred_ground_comparison.gif',
                pred_vs_ground_images, duration=0.5)
