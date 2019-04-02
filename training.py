import time

from LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain import AbstractLearning
from Preprocessors.Abstract_Preprocess import AbstractProcessor
from temp_Graphs.summary import Summary


def train(env: any, learner: AbstractLearning, graph: Summary, processor: AbstractProcessor, episodes: int,
          is_cartpole: bool):

    # =================================================

    # TODO: why is this never used?
    # DISCOUNTED_REWARDS_FACTOR = 0.99

    # ==================================================

    # TODO: see if this works,
    #  make it an optional parameter
    # storing neural network weights and parameters
    # SAVE_MODEL = True
    # LOAD_MODEL = True
    # if LOAD_MODEL == True:
    #     neuralNet.model.save_weights(neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5'))

    # ============================================

    print("\n ==== initialisation complete, start training ==== \n")

    reward_episode = []
    for episode in range(int(episodes)):
        # storing frames as gifs, array emptied each new episode
        episode_frames = []

        observation = env.reset()
        episode_frames.append(observation)
        # Processing initial image cropping, grayscale, and stacking 4 of them
        observation = processor.preprocessing(observation, True)

        start_time = time.time()
        sum_rewards_array = 0  # total rewards for graphing

        # TODO: Why is this never used?
        # game_number = 0  # increases every time a someone scores a point
        game_step = 0  # for discounted rewards, steps for each round
        step = 0  # count total steps for each episode for the graph

        # these arrays are used to calculated and store discounted rewards
        # arrays for other variable are needed for appending to transitions in our learner to work
        # arrays emptied after every round in an episode
        reward_array = []
        if episode % 20 == 0:
            reward_episode = []
        states = []
        actions = []
        next_states = []
        dones = []

        while True:
            # TODO: exception for cartpole
            # if (episode > 150) and (args.environment == 'CartPole-v1'):
            #    env.render()

            # action chooses from  simplified action space without useless keys
            action = learner.choose_action(observation, episode)
            # actions map the simplified action space to the environment action space
            # if action space has no useless keys then action = action_mapped
            action_mapped = processor.mapping_actions_to_keys(action)
            # takes a step
            next_observation, reward, done, _ = env.step(action_mapped)

            episode_frames.append(next_observation)

            # TODO: exception for cartpole
            if is_cartpole:
                # punish if terminal state reached
                if done:
                    reward = -reward

            # appending <s, a, r, s', d> into arrays for storage
            states.append(observation)
            actions.append(action)  # only append the '1 out of 3' action

            reward_array.append(reward)
            sum_rewards_array += reward
            reward_episode.append(sum_rewards_array)

            next_observation = processor.preprocessing(next_observation, False)
            next_states.append(next_observation)
            dones.append(done)

            game_step += 1
            step += 1

            if done:
                # TODO: Why is this never used?
                # avg_reward = np.mean(reward_episode)
                # append each <s, a, r, s', d> to learner.transitons for each game round
                for i in range(game_step):
                    learner.transitions.append((states[i], actions[i], reward_array[i], next_states[i], dones[i]))

                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step)

                # empty arrays after each round is complete
                states, actions, reward_episode, next_states, dones = [], [], [], [], []
                # TODO: see if this works,
                #  then make it optional using parameters (in separate file/function?)
                # record video of environment render
                # env = gym.wrappers.Monitor(env, directory='Videos/' + MODEL_FILENAME + '/', video_callable=lambda
                #    episode_id: True, force=True,write_upon_reset=False)
                break

            observation = next_observation
            # TODO: exception for cartpole
            if is_cartpole:
                # train algorithm using experience replay
                learner.memory_replay(episode)

        # TODO: see if this works,
        #  then make it optional using parameters (in a separate file/function?)
        # make gif
        # if episode != 0 and episode % 5 == 0:
        #     images = np.array(episode_frames)
        #     print('gif = ', len(episode_frames))
        #     print('im = ', len(images))

        #     fname = './gifs/episode'+str(episode)+'.gif'
        #     with imageio.get_writer(fname, mode='I') as writer:
        #         for frame in images:
        #             writer.append_data(frame)

        # if episode != 0 and episode % 5 == 0:
        #     fname = './gifs/episode'+str(episode)+'.gif'
        #     save_frames_as_gif(episode_frames, fname)

        # store model weights and parameters when episode rewards are above a certain amount
        # and after every number of episodes

        # if (SAVE_MODEL == True and episode % 5 == 0):
        #     neuralNet.model.save_weights('./temp_Models/' + MODEL_FILENAME+ 'model.h5', overwrite = True)

        # summarize plots the graph
        graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                        learner.e_greedy_formula)
    # killing environment to prevent memory leaks
    env.close()
