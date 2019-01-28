import gym
from Pong_Brain import Processing
from Pong_Network import *


env = gym.make('Pong-v0')


# Processing.Preproccesing(env.observation_space.shape[:])

nn = neural_net((80, 80, 1), env.action_space.shape[:])
# nn.build_dqn()

for episode in range(1000):
    observation = env.reset()


    for step in range(200):
        env.render()
        
        








