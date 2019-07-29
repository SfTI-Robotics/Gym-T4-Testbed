from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
import keras.backend as K

INPUT_DIM = (84,84,4) # 4 stacked frames
Z_DIM = 32
DENSE_SIZE = 1024
ACTION_DIM = 1
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

ENCODER_DIM = DENSE_SIZE + ACTION_DIM

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 32), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():
     def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.encoder_mu_log_var = self.models[2]
        self.decoder = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE    

    def _build(self):
        # Encoder layers

        action = Input(shape=self.action_dim, name='action_input')          # Input layer for the action, keep separate for calculating reconstruction loss
        vae_action = Dense(self.action_dim, name='action')(action)          # 1
        

        # calculate dimensions after convolution by
        #
        #   h = w = (original w + 2*padding - kernel size) / stride
        #

        
        vae_x = Input(shape=self.input_dim, name='observation_input')       # 84x84x4
        h = Conv2D(32, 6, strides=2, activation='relu')(vae_x)              # 40x40x32
        h = Conv2D(64, 6, strides=2, padding=(2,2) activation='relu')(h)    # 20x20x64
        h = Conv2D(64, 6, strides=2, padding=(2,2) activation='relu')(h)    # 10x10x64
        h = Conv2D(64, 4, strides=2, activation='relu')(h)                  # 4x4x64
        h = Flatten()(h)                                                    # 1024

        h = Concatenate([h, vae_action])                                # 1025

        encoder_h = Dense(ENCODER_DIM, activation='relu')
        z_mean = Dense(self.Z_DIM, name='z_mean')(encoder_h)            # 32
        z_log_var = Dense(self.Z_DIM, name='z_log_var')(encoder_h)      # 32
        z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

        # merge latent space with same action vector that was merged into observation
        zc = Concatenate([z, action])

        # Decoder layers
        decoder = Conv2DTranspose(64, 4, strides=2, activation='relu')(zc)


        return (vae_x)
    
    def train(self, data):
        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE)