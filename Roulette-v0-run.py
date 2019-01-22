import gym

env = gym.make('Roulette-v0')

for i in range(1000):
    state = env.reset()
    for step in range(200):
        env.render()
        
