#! /usr/bin/env python
"""This script handles reading command line arguments and starting the
training process.  It shouldn't be executed directly; it is used by
run_nips.py or run_nature.py.

"""
import os
import time
import argparse
import logging
import cPickle
import numpy as np
import theano
import simplejson as json
import gym 

import experiment
import ale_agent
import q_network
import profile
import pymongo
from ale_utils import ALEPreProcessor
from create_database import create_db


def parameters_as_dict(parameters):
    args_dict = {}
    args = [arg for arg in dir(parameters) if not arg.startswith('_')]
    for arg in args:
        args_dict[arg] = getattr(parameters, arg)
    return args_dict

def save_parameters(args, save_path):
    name = '/'.join((save_path, 'parameters' + '.json'))
    with open(name,'wb') as f:
        json.dump(parameters_as_dict(args), f, sort_keys=True, indent='\t')

def process_args(args, defaults, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--mongo_host', dest="mongo_host",
                        default='localhost',
                        help='mongodb host'
                        '(default localhost)')
   
    parser.add_argument('--mongo_port', dest="mongo_port",
                        type=int, default=27017,
                        help=('mongodb port'))
    parser.add_argument('--env', dest="environment", default=defaults.ENV,
                        help='Problem to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--preprocess', dest="preprocess",
                        action='store_true', default=True,
                        help='preprocess observations')
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                        '(default is the name of the game)')

    parser.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--batch-accumulator', dest="batch_accumulator",
                        type=str, default=defaults.BATCH_ACCUMULATOR,
                        help=('sum|mean (default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--clip-delta', dest="clip_delta", type=float,
                        default=defaults.CLIP_DELTA,
                        help=('Max absolute value for Q-update delta value. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate')
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--phi-length', dest="phi_length",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    parser.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--network-type', dest="network_type",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('nips_cuda|nips_dnn|nature_cuda|nature_dnn' +
                              '|linear (default: %(default)s)'))
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    parser.add_argument('--replay-start-size', dest="replay_start_size",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help=('crop|scale (default: %(default)s)'))
    parser.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                        help='Pickle file containing trained net.')
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str, default=defaults.DEATH_ENDS_EPISODE,
                        help=('true|false (default: %(default)s)'))

    parser.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=defaults.DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))
    parser.add_argument('--cudnn_deterministic', dest="cudnn_deterministic",
                        type=bool, default=defaults.CUDNN_DETERMINISTIC,
                        help=('Whether to use deterministic backprop. ' +
                              '(default: %(default)s)'))

    parser.add_argument('--log_level', dest="log_level",
                        type=str, default=logging.INFO,
                        help=('Log level to terminal. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--progress-frequency', dest="progress_frequency",
                        type=str, default=defaults.PROGRESS_FREQUENCY,
                        help=('Progress report frequency. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--save-path', dest='save_path',
                        type=str, default='../logs')
    parser.add_argument('--profile', dest='profile', action='store_true')

    parameters = parser.parse_args(args)
    if parameters.experiment_prefix is None:
        parameters.experiment_prefix = parameters.environment

    if parameters.death_ends_episode == 'true':
        parameters.death_ends_episode = True
    elif parameters.death_ends_episode == 'false':
        parameters.death_ends_episode = False
    else:
        raise ValueError("--death-ends-episode must be true or false")

    # This addresses an inconsistency between the Nature paper and the Deepmind
    # code. The paper states that the target network update frequency is
    # "measured in the number of parameter updates". In the code it is actually
    # measured in the number of action choices.
    # The default still has the same result as DeepMind's code, only the result 
    # is achieved like DeepMind's paper describes it.
    parameters.freeze_interval = (parameters.freeze_interval //
                                  parameters.update_frequency)

    return parameters



def launch(args, defaults, description):
    """
    Execute a complete training run.
    """

    parameters = process_args(args, defaults, description)
    try:
        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%d-%m-%Y-%H-%M-%S", time.gmtime())
        save_path = parameters.save_path + '/' + parameters.experiment_prefix + time_str 
        os.makedirs(save_path)
    except OSError as ex:
        # Directory most likely already exists
        pass
    try:
        link_path = parameters.save_path + '/last_' + parameters.experiment_prefix
        os.symlink(save_path, link_path)
    except OSError as ex:
        os.remove(link_path)
        os.symlink(save_path, link_path)

    save_parameters(parameters, save_path)
    logger = logging.getLogger()
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # log to file
    fileHandler = logging.FileHandler("{0}/out.log".format(save_path))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    # log to stdout
    import sys
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logFormatter)
    logger.addHandler(streamHandler)
    logger.setLevel(parameters.log_level)

    if parameters.profile:
        profile.configure_theano_for_profiling(save_path)


    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    if parameters.cudnn_deterministic:
        theano.config.dnn.conv.algo_bwd = 'deterministic'



    # TODO make it display 
    if parameters.display_screen:
        pass

    env = gym.make(parameters.environment)
    num_actions = env.action_space.n
    
    if parameters.preprocess:
        preprocessor = ALEPreProcessor(defaults.RESIZED_WIDTH,
                                              defaults.RESIZED_HEIGHT,
                                              parameters.resize_method,)
    else:
        preprocessor = None

    if parameters.nn_file is None:
        network = q_network.DeepQLearner(defaults.RESIZED_WIDTH,
                                         defaults.RESIZED_HEIGHT,
                                         num_actions,
                                         parameters.phi_length,
                                         parameters.discount,
                                         parameters.learning_rate,
                                         parameters.rms_decay,
                                         parameters.rms_epsilon,
                                         parameters.momentum,
                                         parameters.clip_delta,
                                         parameters.freeze_interval,
                                         parameters.batch_size,
                                         parameters.network_type,
                                         parameters.update_rule,
                                         parameters.batch_accumulator,
                                         rng)
        with open(os.path.join(save_path,'net.pkl'),'w') as f:
            cPickle.dump(network,f)
    else:
        handle = open(parameters.nn_file, 'r')
        network = cPickle.load(handle)
        
    client = pymongo.MongoClient(host = parameters.mongo_host,
                                 port = parameters.mongo_port)
                                 
    create_db(parameters.experiment_prefix,
                  host = parameters.mongo_host,
                  port = parameters.mongo_port)
    
    db = client[parameters.experiment_prefix]    
    

    agent = ale_agent.NeuralAgent(db,
                                  network,
                                  parameters.epsilon_start,
                                  parameters.epsilon_min,
                                  parameters.epsilon_decay,
                                  parameters.replay_memory_size,
                                  parameters.replay_start_size,
                                  parameters.update_frequency,
                                  rng, save_path, 
                                  parameters.profile)

    env = gym.make(parameters.environment)
    exp = experiment.GymExperiment(env, agent,preprocessor,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              rng,
                                              parameters.progress_frequency)


    exp.run()



if __name__ == '__main__':
    pass
