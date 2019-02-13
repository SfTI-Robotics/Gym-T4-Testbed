#!/bin/bash

# list the testing algorithm, environment and episode number 
python3 run_main.py -alg DQN -env Pong-v0 -eps 10000
python3 run_main.py -alg DQN -env SpaceInvaders-v0 -eps 10000


# terminate running programs in cmd line 
kill -9 $(jobs -p)