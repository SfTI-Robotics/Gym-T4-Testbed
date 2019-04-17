# Gym-T4-Testbed

- trying to write our own algorithms for gym environments
- setting up benchmark system to run algorithms on various OpenAI environments
----------------
### Dependencies 

- Python 3.6
- Numpy 1.15
- Tensorflow 
- Keras
- OpenAI Gym Atari
- scikit-image
- OpenCV
- matplotlib
- imageio

----------------
### Files Overview

``` bash

# Bash scripts
    evaluate.sh
        using testbed.txt

# Execution scripts 
    run_main.py
        using training.py

# Environment State Preprocessing
    /utils/preprocessing     # folder
        Abstract_Preprocessor.py    # abstract preprocessor class, used in training.py
        implementations for Cartpole, Breakout, MsPacman, Pong, SpaceInvaders

# RL Algorithms
    /agents
        /image_input
            AbstractBrain   # abstract agent class, used in training.py
            implementations for DQN, DoubleDQN, DuelingDQN
        
        /memory
            Memory.py   # storage for replay data
            
        /networks
            dqn_networks.py # build functions for neural networks
            dueling_dqn_networks.py # build functions for neural networks with split layer
            
# Training
    /training
        training.py # trains a RL agent in an environment

# Saving training data
    /utils
        summary.py  # plotting training data
        storing.py  # saving model files and gifs during training process


```



----------------
