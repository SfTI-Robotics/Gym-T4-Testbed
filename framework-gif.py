import gym
from gym import wrappers
# ==============
# from gym_recording.wrappers import TraceRecordingWrapper
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os


env = gym.make('Pong-v0')
# env.monitor.start('/tmp/cartpole-experiment-1')

observation = env.reset()
# env = gym.wrappers.Monitor(env,directory='pong/',force=True,write_upon_reset=True)
for i_episode in range(20):
    observation = env.reset()
#     env = gym.wrappers.Monitor(env,directory='pong/',force=True,write_upon_reset=True)
    for t in range(100):
        env.render()
     #    print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print( "Episode finished after {} timesteps".format(t+1))
            env = gym.wrappers.Monitor(env,directory='pong/',force=True,write_upon_reset=True)
            break

env.env.close()

# env = gym.make('CartPole-v0')
# env = Monitor(env, './video', force = True)
# observation = env.reset()
# cum_reward = 0
# frames = []


# env = gym.make('FrozenLake-v0')
# # env = Monitor(env, './video')
# # video = VideoRecorder(env)
# env.monitor.start('/vid/cartpole-experiment-1')
# try:
#      env.reset()
#      # video.capture_frame()
#      # video.close()
# finally:
     # os.remove(video.path)


# rec = video_recorder(env, )




# for t in range(10):
#     # Render into buffer.
#     # You will still see the window.
#     env.render()
#     frames.append(env.render(mode = 'rgb_array'))
# #     env = TraceRecordingWrapper(env)





#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)

#     if done:
#      break

