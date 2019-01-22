
import sys
import gym
from gym import wrappers


# Change the the gym environments into the Command line inputs
# sys.argv is a list of commandline argumenets

#Making enum to make it readable in the code
ENVIRONMENT = sys.argv[1]
EPISODES = int(sys.argv[2])
STEPS = int(sys.argv[3])

env = gym.make(ENVIRONMENT)

# 'Copy-v0' , Place command line argument when running the file
#Providing 3 command line arugmenets (Env, Episodes , Steps)
env = wrappers.Monitor(env, '/tmp/' + ENVIRONMENT, force=True)


for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(STEPS):
        action = (1,1,observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.render()