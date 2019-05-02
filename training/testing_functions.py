import datetime
import numpy as np

from utils.summary import Summary


def test(learner, env, filename, path, config, processor, min_reward=None, max_reward=None):

    test_summary = Summary(['sumiz_step', 'sumiz_reward'],
                           name=filename + str(datetime.datetime.now()),
                           save_path=path,
                           min_reward=min_reward,
                           max_reward=max_reward)
    test_rewards = []
    test_steps = []

    for e in range(100):
        test_state = env.reset()

        test_sum = 0
        test_step = 0
        while True:
            test_action = learner.choose_action(test_state)
            test_next_state, test_reward, test_done, _ = env.step(test_action)
            if config['environment'] == 'CartPole-v1':
                # punish if terminal state reached
                if test_done:
                    test_reward = -test_reward
            test_next_state = processor.preprocessing(test_next_state, False)
            test_state = test_next_state

            test_sum += test_reward
            test_step += 1

            if test_done:
                print('Completed Episode = ' + str(e), ' epsilon =', "%.4f" % learner.epsilon,
                      ', steps = ', test_step,
                      ", total reward = ", test_sum)
                test_rewards.append(test_sum)
                test_steps.append(test_step)
                break

    test_summary.summarize(step_counts=test_steps, reward_counts=test_rewards)
    print('Sum Rewards = ', np.sum(test_rewards),
          ' Mean Reward = ', np.mean(test_rewards),
          ' Median Rewards = ', np.median(test_rewards),
          ' Standard Deviation = ', np.std(test_rewards))