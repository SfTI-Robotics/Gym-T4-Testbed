# https://blog.csdn.net/storMouse/article/details/52223688
# !/usr/bin/python

import numpy as np
import gym
import random

env = gym.make('Copy-v0')

# observation_space: Discrete(6)
# action_space: (Discrete(2), Discrete(2), Discrete(5))

# states: 0, 1, 2, 3, 4, 5

# MARK: problem specific functions


def action_to_index(action):
    return action[0]*10+action[1]*5+action[2]


def index_to_action(index):
    action = [index/10, (index/5) % 2, index % 5]
    return tuple(action)


# MARK: Monte Carlo ES method

STATE_COUNT = 6
ACTION_COUNT = 20


def init_mces(state_count, action_count):
    q = np.random.rand(state_count, action_count)
    rets = np.zeros((state_count, action_count), dtype=np.double)
    policy = [random.randint(0, action_count-1) for _ in range(state_count)]
    return q, rets, policy


def learning(env):
    q, rets, policy = init_mces(STATE_COUNT, ACTION_COUNT)
    gamma = 0.7
    epsilon = 1
    total_score = 0.0
    i_episode = 0

    for i_episode in range(5000):
        total_reward = 0
        observation = env.reset()
        g = np.zeros((STATE_COUNT, ACTION_COUNT), dtype=np.double)
        passed = np.zeros((STATE_COUNT, ACTION_COUNT), dtype=np.double)

        for t in range(100):
            raw_action = policy[observation]
            # 1 - epsilon greedy
            if random.random() < epsilon:
                raw_action = action_to_index((random.randint(0, 1),
                                              random.randint(0, 1),
                                              random.randint(0, 4)))
            if passed[observation, raw_action] == 0.0:
                passed[observation, raw_action] = 1.0

            action = index_to_action(raw_action)
            # env.render()
            observation, reward, done, info = env.step(action)
            reward -= 0.5  # IMPORTANT: punish useless (even dangerous) actions whose environment reward is 0.0
            total_reward += reward

            for i in range(STATE_COUNT):
                for j in range(ACTION_COUNT):
                    passed[i][j] *= gamma
                    g[i][j] += reward * passed[i][j]
            if done:
                break

        # reduce exploration chance
        if i_episode % 100 == 0:
            epsilon *= 0.9

        rets += g
        q = rets / (i_episode+1)
        policy = np.argmax(q, axis=1).tolist()

        total_score += total_reward

    return policy, i_episode


def test_policy(env, policy):
    total_reward = 0.0
    obs = env.reset()
    for t in range(1000):
        action = index_to_action(policy[obs])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print('total reward: %f'%total_reward)


# env.monitor.start('Copyv0-experiment-0')
policy, n_episode = learning(env)
# env.monitor.close()

print('final policy: '+str(policy))
print('episodes trained: '+str(n_episode))
test_policy(env, policy)

