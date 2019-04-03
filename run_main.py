"""
this is the universal run script for all environments

"""
# import dependencies
import argparse
from argparse import RawTextHelpFormatter
import sys
import gym
import datetime

# for graphing
from summary import Summary
from training import train


# TODO: reduce length of functions wherever possible
# TODO: add method comments

if __name__ == "__main__":

    # For more on how argparse works see documentation
    # create argument options
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-alg", "--algorithm",
                        help="select a algorithm: \n QLearning \n DQN \n DoubleDQN \n DuellingDQN \n DDDQN")
    parser.add_argument("-env", "--environment",
                        help="select a environment: \n Pong-v0 \n SpaceInvaders-v0 \n MsPacman-v0")
    parser.add_argument("-eps", "--episodes", help="select number of episodes to graph")
    parser.add_argument("-save", "--save_model", help='saves model and tensorboard summary regularly',
                        action='store_true')
    parser.add_argument("-load", "--load_model", help='loads model for given algorithm and environment',
                        action='store_true')

    # retrieve user inputted args from cmd line
    args = parser.parse_args()
    is_cartpole = (args.environment == 'CartPole-v1')

    # Prepossessing folder
    # this takes care of the environment specifics and image processing
    if args.environment == 'Pong-v0':
        import Preprocessors.Pong_Preprocess as Preprocess
        print('Pong works')
    elif args.environment == 'SpaceInvaders-v0':
        import Preprocessors.SpaceInvaders_Preprocess as Preprocess
        print('SpaceInvaders works')
    elif args.environment == 'MsPacman-v0':
        import Preprocessors.MsPacman_Preprocess as Preprocess
        print('MsPacman works')
    elif args.environment == 'Breakout-v0':
        import Preprocessors.Breakout_Preprocess as Preprocess
        print('Breakout works')
    elif args.environment == 'CartPole-v1':
        import Preprocessors.Cartpole_Preprocess as Preprocess
        print('Cartpole works')
    else:
        sys.exit("Environment not found")

    # create gym env
    env = gym.make(args.environment)
    # initialise processing class specific to environment
    processor = Preprocess.Processor()
    # state space is determined by the deque storing the frames from the env
    state_space = processor.get_state_space()

    if is_cartpole:
        state_space = env.observation_space.shape[0]

    # action space given by the environment
    action_space = env.action_space.n

    # ============================================

    # here we change the action space if it contains 'useless' keys or actions that do the same thing
    # if no useless keys it just returns the envs defined action space
    # This function is created in the preprocess file
    action_space = processor.new_action_space(action_space)

    # algorithm folder
    if args.algorithm == 'DQN':
        from LearningAlgorithms.DQN.Brain import Learning
        print('DQN works')
    elif args.algorithm == 'DoubleDQN':
        from LearningAlgorithms.Double_DQN.Brain import Learning
        print('Double works')
    else:
        sys.exit("Algorithm not found")

    learner = Learning(state_space, action_space, is_cartpole)

    # ============================================

    # Graphing results
    now = datetime.datetime.now()
    MODEL_FILENAME = args.environment + '_' + args.algorithm + '_'
    # our graphing function
    # summary sets the ranges and targets and saves the graph
    graph = Summary(summary_types=['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_average_reward', 'sumiz_epsilon'],
                    # maximum exploitation value
                    epsilon_goal=0.99,
                    # desired name for file
                    name=MODEL_FILENAME + str(now),
                    # file path to save graph. i.e "/Desktop/Py/Scenario_Comparision/Maze/Model/"
                    # SAVE_PATH = "/github/Gym-T4-Testbed/Gym-T4-Testbed/temp_Graphs/",
                    save_path="/Gym-T4-Testbed/temp_Graphs/",
                    # episode upper bound for graph
                    episode_max=int(args.episodes),
                    # step upper bound for graph
                    step_max_m=processor.step_max,
                    # time upper bound for graph
                    time_max_m=processor.time_max,
                    # reward upper bound for graph
                    reward_min_m=processor.reward_min,
                    # reward lower bound for graph
                    reward_max_m=processor.reward_max)

    # train learner and plot results
    train(env, learner, graph, processor, int(args.episodes), MODEL_FILENAME[:-1], is_cartpole,
          save_model=args.save_model, load_model=args.load_model, gif=False)
