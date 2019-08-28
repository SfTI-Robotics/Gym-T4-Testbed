#!/bin/bash 

ENV='Breakout-v0'

python generate_data.py --env_name $ENV && python train_cvae.py --new_model --env_name $ENV