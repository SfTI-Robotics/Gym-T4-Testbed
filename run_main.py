"""
this is the universal run script for all environments

"""
# print("enter -h for options you can enter")

import argparse
import sys
import numpy as np
import time
import gym

from argparse import RawTextHelpFormatter

from summary import summary
import time
import datetime
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-algorithm", help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
parser.add_argument("-environment", help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
# add global variables
parser.add_argument("-episodes", help="select number of episodes to graph")
# parser.add_argument("-steps", help="select number of steps")

args = parser.parse_args()

# Preprocessing folder
if args.environment == 'Pong-v0':
    import Preprocess.Pong_Preprocess as preprocess
    print('Pong works')
elif args.environment == 'SpaceInvaders-v0':
    import Preprocess.SpaceInvaders_Preprocess as preprocess
    print('SpaceInvaders works')
elif args.environment == 'MsPacman-v0':
    import Preprocess.MsPacman_Preprocess as preprocess
    print('MsPacman works')
else :

    print("Environment not found")

# # Brain and Network folder
if args.algorithm == 'QLearning':
    import Q_table.Brain as brain
    # import Q_table.Network as network
    print('Q tables work')
elif args.algorithm == 'DQN':
    import DQN.Brain as brain
    # import DQN.Network as network
    print('DQN works')
elif args.algorithm == 'DoubleDQN':
    import Double_DQN.Brain as brain
    # import Double_DQN.Network as network
    print('Double works')
elif args.algorithm == 'DuellingDQN':
    import Dueling_DQN.Brain as brain
    # import Dueling_DQN.Network as network
    print('Dueling works')
elif args.algorithm == 'DDDQN':
    import DDDQN_PER.Brain as brain
    # import DDDQN_PER.Network as network
    print('PER works')
else :
    print("Algorithm not found")
# ============================================
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_FILENAME = args.environment + '_' + args.algorithm
# ============================================

env = gym.make(args.environment)

# initialise objects
processor = preprocess.Processing()
state_space = processor.get_state_space()
action_space = env.action_space.n 
action_space=processor.new_action_space(action_space)


# print('state_space: ', env.observation_space.shape)

# neuralNet = network.neural_net(state_space, action_space)
# learner = brain.Learning(state_space, action_space, neuralNet)
learner = brain.Learning(state_space, action_space)

# if LOAD_MODEL == True:
#     neuralNet.model.save_weights(neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5'))

# Graphing results
now = datetime.datetime.now()
graph = summary(summary_types = ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_epsilon'],
            # the optimal step count of the optimal policy
            step_goal = 0,
            # the maximum reward for the optimal policy
            reward_goal = 0,
            # maximum exploitation value
            epsilon_goal = 0.99,
            # desired name for file
            NAME = MODEL_FILENAME + str(now),
            # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
            # SAVE_PATH = "/github/Gym-T4-Testbed/Gym-T4-Testbed/temp_Graphs/",
            SAVE_PATH = "/Gym-T4-Testbed/temp_Graphs/",
            EPISODE_MAX = int(args.episodes),

            STEP_MAX_M = processor.step_max,

            TIME_MAX_M = processor.time_max,

            REWARD_MIN_M = processor.reward_min,

            REWARD_MAX_M = processor.reward_max
    )
print("\n ==== initialisation complete, start training ==== \n")

# ==================================================
#
for episode in range(500):
    episode_transitions = []
    
    observation = env.reset()
    observation = processor.Preprocessing(observation, True)

    start_time = time.time()
    episode_rewards = 0#total rewards
    
    #these arrays and variables are used to calculated and store discounted rewards
    game_number = 0 # increases every time a someone scores a point
    game_step = 0 #for discounted rewards
    
    # arrays emptied after every round in an episode 
    reward_array=[]#for d_r
    states=[]
    actions=[]
    next_states=[]
    dones=[]

    

    while True:
        env.render()

        action = learner.choose_action(observation, episode)#action chooses from 3/3
        action_mapped = processor.mapping_actions_to_keys(action) # actions returns to length 3/6 but can only choose 0,2,3
        # print('action =',action)
        next_observation, reward, done, _ = env.step(action_mapped)
        episode_rewards += reward
        reward_array.append(reward)
        states.append(observation)
        actions.append(action)
        dones.append(done)

        ## Done s1 - s5 reward -1
        ## Iterate s5- s1 s5 -> -1, s4 -> %-1 ... etc

        # rewards should be set up as a 1d array
        # gamma as a float 
        # these could be used as inputs to a function
        
        # discounted rewards could be set up in preprocessing 
        # so our run main doesn't change in envs that don;t need it dis_rew=rewards

        # In Pong rewards can be {-1,0,1}
        # when -1 or 1 game has been reset
        # have a running add variable that resets to 0 whenever this happens

        # np.zeroes_like(rewards) creates an array the same shape as the input

        next_observation = processor.Preprocessing(next_observation, False)
        next_states.append(next_observation)
        #append episode_transitions after dicounted rewards has been calculated
        # perhaps create a function for it

        #episode_transitions.append((observation, action, reward, next_observation, done))
        ## print(np.shape(episode_transitions))

        

        # if episode > 1:
        #     learner.memory_replay()
        
        game_step += 1
        #episode_transitions = list(episode_transitions)
#########################################################################################################
        if (not reward == 0) or (done) :
            print(  'game_number =',   game_number , 'game_step = ', game_step)
            #print('reward =', episode_rewards, 'steps =', game_step)
            ## call discounted function here 
            reward_array=processor.discounted_rewards(reward_array,learner.gamma)
            #print('reward array= \n', reward_array)
                
            #append to leraner.transitons

            for i in range(game_step):
                learner.transitions.append((states[i], actions[i], reward_array[i],next_states[i],dones[i]))


            states, actions, reward_array,next_states,dones  = [], [], [], [], []
            game_number += 1
            game_step=0
            
            if done:
                print('Completed Episode ' + str(episode))
                game_number = 0
    
                # learner.transitions.extend(episode_transitions)
                learner.memory_replay()

                break

        


#########################################################################################################
        # if not reward == 0:
        #     game_step += 1  
        #     for i in reversed(range(step)):
        #         print(episode_transitions[i][2])
        #         #  find reward of observations before it, change it based on a percentage
        #         episode_transitions[i][2] += reward * 0.2 ** (1/i)
        # #call the memory replay function to learn at the end of every episode
        # episode_transitions = tuple(episode_transitions)
##########################################################################################################
        

        observation = next_observation

    #observation -> [UP, DOWN, NO_OP]

    # store model weights and parameters when episode rewards are above a certain amount 
    # and after every number of episodes
    #  === change reward threshold
    
    # if (SAVE_MODEL == True and episode % 5 == 0):
    #     neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5', overwrite = True)

    graph.summarize(episode, game_step, time.time() - start_time, episode_rewards, learner.epsilon)

env.close()
