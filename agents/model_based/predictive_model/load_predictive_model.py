import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from predictive_model import AutoEncoder

def load_predictive_model(env_name, n_actions):
    # weights = 'predictive_model/cvae_weights_mult_%s.h5' % env_name
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(parent_dir, 'models/predictive_model_weights_%s.h5' % env_name)
    model = AutoEncoder(n_actions)
    model.set_weights(weights_path)

    return model