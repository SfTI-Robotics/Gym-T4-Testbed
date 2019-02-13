import gym
from gym import wrappers
# ==============
# from gym_recording.wrappers import TraceRecordingWrapper
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os


env = gym.make('Pong-v0')

observation = env.reset()
for i_episode in range(20):
   observation = env.reset()
   for t in range(3000):
      env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      if done:
         print( "Episode finished after {} timesteps".format(t+1))
         # env = gym.wrappers.Monitor(env,directory='Videos/pong/',video_callable=lambda episode_id: True, force=True,write_upon_reset=True)
         env = gym.wrappers.Monitor(env,directory='Videos/pong/',video_callable=lambda episode_id: True, force=True,write_upon_reset=False)
         break

env.env.close()