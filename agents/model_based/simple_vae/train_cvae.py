from simple_vae import CVAE
import argparse
import numpy as np
import os

DIR_NAME = './data/rollout/'
M=300

SCREEN_SIZE_X = 84
SCREEN_SIZE_Y = 84
ACTION_SPACE_X = 1
ACTION_SPACE_Y = 4


def import_data(N):
  filelist = os.listdir(DIR_NAME)
  filelist.sort()
  length_filelist = len(filelist)

  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  observation = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 4), dtype=np.float32)
  action = np.zeros((M*N, ACTION_SPACE_Y), dtype=np.float32)
  next_frame = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 1), dtype=np.float32)

  idx = 0
  file_count = 0

  for file in filelist:
    try:
        obs_data = np.load(DIR_NAME + file)['obs']
        action_data = np.load(DIR_NAME + file)['actions']
        next_data = np.load(DIR_NAME + file)['next_frame']

        observation[idx:(idx + M), :, :, :] = obs_data
        action[idx:(idx + M), :] = action_data
        next_frame[idx:(idx + M), :, :, :] = next_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
            print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
    except:
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

  return [observation,action,next_frame], N

def main(args):

    new_model = args.new_model
    N = int(args.N)
    epochs = int(args.epochs)

    cvae = CVAE()

    if not new_model:
        try:
            cvae.set_weights('./vae_weights.h5')
        except:
            print("Either set --new_model or ensure ./models/weights.h5 exists")
            raise

    try:
        data, N = import_data(N)
    except:
        print('NO DATA FOUND')
        raise
        
    print('DATA SHAPE = {}'.format(data[0].shape))

    for epoch in range(epochs):
        print('EPOCH ' + str(epoch))
        cvae.train(data[0],data[1],data[2])
        cvae.save_weights('./cvae_weights.h5')
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  args = parser.parse_args()

  main(args)