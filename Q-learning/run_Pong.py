import gym
from Pong_Brain import *
from Pong_Network import *


env = gym.make('Pong-v0')

observation_space = env.observation_space.shape[:]
action_space = env.action_space.shape[:]

# initialise 
processor = Processing()
learner = Learning(action_space)



for episode in range(1000):
    observation = env.reset()
    observation = processor.four_frames_to_state(observation)

    for step in range(200):
        env.render()
        
        action= learner.choose_action(observation)
        
        next_observation, reward, done, _ = env.step(action)
        next_observation = processor.four_frames_to_state(next_observation)
        learner.transitions.append([observation], action, reward, [next_observation])
                    
        if done:
            #call the memory replay function to learn at the end of every episode
            learner.memory_replay()                                        
            break
                
        observation = next_observation









