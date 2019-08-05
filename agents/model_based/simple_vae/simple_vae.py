from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

INPUT_DIM = (84,84,4) # 4 stacked frames
Z_DIM = 32
DENSE_SIZE = 1024
ACTION_DIM = 4
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5
BATCH_SIZE = 100

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 32), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class CVAE():
    def __init__(self):
        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE   
        self.batch_size = BATCH_SIZE
        
        self.model = self._build() 

    def _build(self):
                
        vae_x = Input(shape=self.input_dim, name='observation_input')       # 84x84x4
        
        ####################################################################
        #                        Step 1: CONVOLUTION                       #
        ####################################################################

        
        h = Conv2D(32, 6, strides=2, activation='relu')(vae_x)              # 40x40x32
        h1 = Conv2D(64, 6, strides=2, activation='relu')(h)                 # 18x18x64
        h2 = Conv2D(128, 6, strides=2, activation='relu')(h1)               # 7x7x128
        h3 = Conv2D(256, 4, strides=2, activation='relu')(h2)               # 2x2x256
        h4 = Flatten()(h3)                                                  # 1024


        ####################################################################
        #                     Step 2: ACTION ENCODING                      #
        ####################################################################


        action_input = Input(shape=(self.action_dim,), name='action_input')    # 4
        # encoded = Concatenate()([h4, action_input])                     # 1028


        ####################################################################
        #                  Step 3: LATENT SPACE ENCODING                   #
        ####################################################################
        

        # encoder_h = Dense(ENCODER_DIM, activation='relu')()
        z_mean = Dense(self.z_dim, name='z_mean')(h4)                      
        z_log_var = Dense(self.z_dim, name='z_log_var')(h4)                
        z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

        # # merge latent space with same action vector that was merged into observation
        zc = Concatenate(axis=-1)([z, action_input])

        
        ####################################################################
        #                     Step 4: DECONVOLUTION                        #
        ####################################################################


        decoder_dense = Dense(DENSE_SIZE, name='decoder_input')(zc)
        decoder_reshape = Reshape((1,1,1024), name='unflatten')(decoder_dense)
        decoder = Conv2DTranspose(128, 7, strides=2, activation='relu')(decoder_reshape)
        decoder_2 = Conv2DTranspose(64, 6, strides=2, activation ='relu')(decoder)
        decoder_3 = Conv2DTranspose(32, 6, strides=2, activation ='relu')(decoder_2)
        # decoder_4 = Conv2DTranspose(32, 6, strides=2, activation ='relu')(decoder_3) 
        decoder_out = Conv2DTranspose(1, 6, strides=2, activation='sigmoid')(decoder_3)
        
        vae_full = Model([vae_x,action_input],decoder_out)


        ####################################################################
        #                        LOSS FUNCTIONS                            #
        ####################################################################

        def vae_r_loss(y_true, y_pred):
            r_loss = K.sum(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss

        def vae_kl_loss(y_true, y_pred):

            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = 1)
            kl_loss = K.maximum(kl_loss, KL_TOLERANCE * Z_DIM)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)


        ####################################################################
        #                          Step 5: TRAINING                        #
        ####################################################################
        
        opti = Adam(lr=LEARNING_RATE)
        vae_full.compile(optimizer=opti, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

        print(vae_full.summary())
        return (vae_full)
    
    def train(self, obs, action, next_frame):
        input_data = [obs, action]
        self.model.fit(x=input_data, y=next_frame,
                shuffle=True,
                epochs=1,
                batch_size=self.batch_size)
    
    def set_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def predict(self, observation, action):
        return self.model.predict([observation,action])
    
    def print_layer_shapes(self):
        self.model.summary()
        