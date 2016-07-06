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
from agents import dqn_agent
import lasagne
from network import q_network
from network.parameter_server import ParameterServer
import profile
from datasets import tabular_dataset
import pymongo
from network import network_handler
from network.network_trainer import NetworkTrainer
from utils.ale_utils import ALEPreProcessor
from utils.database import create_db
import multiprocessing
from multiprocessing.managers import SyncManager


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
    ##data set parameters
    data_group = parser.add_argument_group('database', 'data set parameters')

    data_group.add_argument('--database', dest="database",
                        default='ndarray',
                        help='replay database type:\
                        ndarray, mongodb or none (default: ndarray)'
                        '(default localhost)') 
    data_group.add_argument('--mongo-host', dest="mongo_host",
                        default='localhost',
                        help='mongodb host'
                        '(default localhost)')  
    data_group.add_argument('--mongo-port', dest="mongo_port",
                        type=int, default=27017,
                        help=('mongodb port'))
    data_group.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    ##parameter server  parameters
    param_group = parser.add_argument_group('database', 'data set parameters')

    param_group.add_argument('--param-host', dest="param_host",
                        default='localhost',
                        help='parameter server host'
                        '(default localhost)')  
    param_group.add_argument('--param-port', dest="param_port",
                        type=int, default=50000,
                        help=('parameter server port'))
    param_group.add_argument('--password', dest="password",
                        type=str, default='NO_MORE_SECRETS',
                        help=('parameter server password'))
    param_group.add_argument('--max_delay', dest="max_delay",
                        type=int, default=10,
                        help=('max update steps before gradients are stale'))
    ##experiment  parameters
    exp_group = parser.add_argument_group('experiment', 'experiment parameters')
    exp_group.add_argument('--env', dest="environment", default=defaults.ENV,
                        help='Problem to run (default: %(default)s)')
    exp_group.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    exp_group.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    exp_group.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    exp_group.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    exp_group.add_argument('--preprocess', dest="preprocess",
                        action='store_true', default=True,
                        help='preprocess observations')
    exp_group.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                        '(default is the name of the game)')
                        
    exp_group.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help=('crop|scale (default: %(default)s)'))
    exp_group.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=defaults.DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))
    ###network parameters
    net_group = parser.add_argument_group('network', 'neural network parameters')
    net_group.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    net_group.add_argument('--batch-accumulator', dest="batch_accumulator",
                        type=str, default=defaults.BATCH_ACCUMULATOR,
                        help=('sum|mean (default: %(default)s)'))
    net_group.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    net_group.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    net_group.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    net_group.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    net_group.add_argument('--clip-delta', dest="clip_delta", type=float,
                        default=defaults.CLIP_DELTA,
                        help=('Max absolute value for Q-update delta value. ' +
                              '(default: %(default)s)'))
    net_group.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate')
    net_group.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    net_group.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    net_group.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    net_group.add_argument('--phi-length', dest="phi_length",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    net_group.add_argument('--network-type', dest="network_type",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('nips|nips|nature' +
                              '|linear (default: %(default)s)'))
    net_group.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    net_group.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    net_group.add_argument('--replay-start-size', dest="replay_start_size",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' +
                              '(default: %(default)s)'))

    net_group.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                        help='Pickle file containing trained net.')
    net_group.add_argument('--cudnn_deterministic', dest="cudnn_deterministic",
                        type=bool, default=defaults.CUDNN_DETERMINISTIC,
                        help=('Whether to use deterministic backprop. ' +
                              '(default: %(default)s)'))

    #top level parameters

    

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
    parser.add_argument('--mode', dest="launch_mode",
                        type=str, default='local',
                        help=('type of process to launch local,\
                        ,training,param_server distributed or async\
                        (default:local)'))
    
    parser.add_argument('--num_agents', dest="n_agents",
                        type=int, default=2,
                        help=('number of agents, only for async mode\
                        (default:2)'))
    

    parameters = parser.parse_args(args)
    if parameters.experiment_prefix is None:
        parameters.experiment_prefix = parameters.environment


    # This addresses an inconsistency between the Nature paper and the Deepmind
    # code. The paper states that the target network update frequency is
    # "measured in the number of parameter updates". In the code it is actually
    # measured in the number of action choices.
    # The default still has the same result as DeepMind's code, only the result 
    # is achieved like DeepMind's paper describes it.
    parameters.freeze_interval = (parameters.freeze_interval //
                                  parameters.update_frequency)

    return parameters

def run_exp(exp):
    exp.run()

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
        
        
    if preprocessor is None:
        obs_space = env.observation_space
        if isinstance(obs_space,gym.spaces.Box):   
            obs_shape = obs_space.high.shape
        elif isinstance(obs_space,gym.spaces.Discrete):
            obs_shape = (1,)
        else:
            #TODO: handle tuple spaces
            raise RuntimeError('observation space not supported')
        
    else:
        obs_shape = preprocessor.output_shape

    if parameters.nn_file is None:
        network = q_network.DeepQLearner(obs_shape,
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
        
   # client = pymongo.MongoClient(host = parameters.mongo_host,
   #                              port = parameters.mongo_port)
                                 
   # create_db(parameters.experiment_prefix,
   #               host = parameters.mongo_host,
   #               port = parameters.mongo_port)
    
  #  db = client[parameters.experiment_prefix]    
    
    #env = gym.make(parameters.environment)
    
    if parameters.launch_mode == 'agent':
        db_size = max(parameters.update_frequency,parameters.batch_size)
    else:
        db_size = parameters.replay_memory_size

    
    training_dataset = tabular_dataset.DataSet(rng,obs_shape,
                                      obs_type='uint8',
                                      act_type='uint8',
                                      max_steps=db_size , 
                                      phi_length=parameters.phi_length)
                                      
    test_dataset = tabular_dataset.DataSet(rng, obs_shape,
                                      obs_type='uint8',
                                      act_type='uint8',
                                      max_steps=5000, 
                                      phi_length=parameters.phi_length)

# Experiments can be run in several modes:
#   -Local: standard dqn agent keeps database of samples and locally trains
#           deep q-network using batch sampling
#
#  - distributed: gorilla-like setup, multiple agents act in their own 
#           environments. Parameters are shared through parameter servers
#           Agents do not train own netowork. This is done in separate 
#           process(es). Data can be local to agents or shared through a 
#           common replay database
#
#   - trainer: runs a batch training agent. Samples batches from dataset and
#           updates network parameters. Parameters are shared true parameter
#           server. Does not run actor that interacts with environment.
#
# - asynchronous: async RL like training. Multiple agents interact with their
#           own environment. Locally train parameters, but share these through
#           parameter server. Agents do not us replay data but update using 
#           last k samples. Typically assumes agents are running on same host,
#           but remote sharing is possible (but incurs communication overhead).
#                                      
                                      
    if parameters.launch_mode== 'local':
        #create local trainer
        net_handler = network_handler.NetworkHandler(
                                      network,
                                      training_dataset
                                      )
        #create agent
        agent = dqn_agent.NeuralAgent(training_dataset,
                                  test_dataset,
                                  net_handler,
                                  parameters.epsilon_start,
                                  parameters.epsilon_min,
                                  parameters.epsilon_decay,
                                  parameters.replay_start_size,
                                  parameters.update_frequency,
                                  rng, save_path, 
                                  parameters.profile)
        #create experiment
        exp = experiment.GymExperiment(env, agent,preprocessor,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              rng,
                                              parameters.progress_frequency)

        #run experiment
        exp.run()
        
    elif parameters.launch_mode == 'actor':
        #create parameter client
        net_handler = network_handler.RemoteNetworkHandler(
                            network,
                            host = parameters.param_host,
                            port = parameters.param_port,
                            authkey = parameters.password)
        

        #create agent                             
        agent = dqn_agent.NeuralAgent(training_dataset,
                                  test_dataset,
                                  net_handler,
                                  parameters.epsilon_start,
                                  parameters.epsilon_min,
                                  parameters.epsilon_decay,
                                  parameters.replay_start_size,
                                  parameters.update_frequency,
                                  rng, save_path, 
                                  parameters.profile)
        #create experiment
        exp = experiment.GymExperiment(env, agent,preprocessor,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              rng,
                                              parameters.progress_frequency)


        exp.run()
    elif parameters.launch_mode == 'param_server':
        #create shared parameters
        ns = ParameterServer(lasagne.layers.helper.get_all_params(network.l_out),
                             parameters.update_rule,
                             parameters.learning_rate,
                             parameters.rms_decay,
                             parameters.rms_epsilon,
                             parameters.momentum,
                             parameters.freeze_interval,
                             max_delay = parameters.max_delay)
        
        #setup parameter server
        class ParamManager(SyncManager): pass
        ParamManager.register('get_space', callable=lambda: ns)

        manager = ParamManager(address=(parameters.param_host,
                                       parameters.param_port),
                            authkey = parameters.password)
        logger.info('launching server')
        manager.get_server().serve_forever()
        
    elif parameters.launch_mode == 'learner':

        
        #setup parameter client
        net_handler = network_handler.AsyncNetworkHandler(
                            network,
                            host = parameters.param_host,
                            port = parameters.param_port,
                            authkey = parameters.password,
                            dataset= training_dataset,
                            batch_size = parameters.batch_size)
        
        #launch trainer
        trainer = NetworkTrainer(net_handler,                            
                            min_samples = parameters.replay_start_size,
                            max_updates = 
                            parameters.steps_per_epoch*parameters.epochs
                            )
    

        trainer.do_training()
        
    elif parameters.launch_mode == 'agent':
        
        #create shared parameters
        ns = ParameterServer(lasagne.layers.helper.get_all_params(network.l_out),
                             parameters.update_rule,
                             parameters.learning_rate,
                             parameters.rms_decay,
                             parameters.rms_epsilon,
                             parameters.momentum,
                             parameters.freeze_interval,
                             max_delay = parameters.max_delay)
        
        #setup parameter server
        class ParamManager(SyncManager): pass
        ParamManager.register('get_space', callable=lambda: ns)

        manager = ParamManager(address=(parameters.param_host,
                                       parameters.param_port),
                            authkey = parameters.password)
        #server = manager.get_server()
    
        logging.info('starting parameter server')
        
        manager.start()
        #launch server
        procs = []

        #p = multiprocessing.Process(target=lambda: server.serve_forever())
        #procs.append(p)
        #server.serve_forever()
        
        lock = manager.Lock()
        #create agent processes
        #NOTE: async updating uses multiprocessing, not multithreading

        logging.info('launching agent processes')


        for a in xrange(parameters.n_agents):
            #create parameter client
            net_handler = network_handler.AsyncNetworkHandler(
                            network, lock,
                            host = parameters.param_host,
                            port = parameters.param_port,
                            authkey = parameters.password,
                            dataset= training_dataset,
                            batch_size = parameters.batch_size)
            
            #create local environment
            env = gym.make(parameters.environment)

            a_path = os.path.join(save_path,'agent'+str(a))
            try:
                os.makedirs(a_path)
            except OSError as ex:
                # Directory most likely already exists
                pass
            #use different random seeds for agents
            #if not parameters.deterministic:
            rng = np.random.RandomState()
        
            agent = dqn_agent.NeuralAgent(training_dataset,
                                  test_dataset,
                                  net_handler,
                                  parameters.epsilon_start,
                                  parameters.epsilon_min,
                                  parameters.epsilon_decay,
                                  1, #don't wait for replay data
                                  parameters.update_frequency,
                                  rng, a_path, 
                                  parameters.profile)

            exp = experiment.GymExperiment(env, agent,preprocessor,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              rng,
                                              parameters.progress_frequency)
            p = multiprocessing.Process(target=run_exp, args=(exp,))
            procs.append(p)
            p.start()
            time.sleep(1) #decreases probability of overwriting updates
            
        #wait for all agents to finish
        for p in procs:
            p.join()
        
    else:
        raise RuntimeError('unknown launch mode: '+
                            parameters.launch_mode)

        



if __name__ == '__main__':
    pass
