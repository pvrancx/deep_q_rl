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
import simplejson as json


import pymongo
from mongo_dataset import MongoDataset
from param_server import ParameterServer
from network_trainer import NetworkTrainer



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
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')
   
    parser.add_argument('--mongo_port', dest="mongo_port",
                        type=int, default=27017,
                        help=('mongodb port'))
    parser.add_argument('--phi-length', dest="phi_length",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of SGD updates before params are uploaded to db. '+
                              '(default: %(default)s)'))
    parser.add_argument('--replay-start-size', dest="replay_start_size",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                        '(default is the name of the game)')
    
    parser.add_argument('--nn-file', dest="nn_file", type=str, default='net.pkl',
                        help='Pickle file containing trained net.')
    

    parser.add_argument('--log_level', dest="log_level",
                        type=str, default=logging.INFO,
                        help=('Log level to terminal. ' +
                              '(default: %(default)s)'))

    parser.add_argument('--save-path', dest='save_path',
                        type=str, default='../logs')
    parser.add_argument('--profile', dest='profile', action='store_true')

    parameters = parser.parse_args(args)
    if parameters.experiment_prefix is None:
        name = os.path.splitext(os.path.basename(parameters.rom))[0]
        parameters.experiment_prefix = name

   


    return parameters



def launch(args, defaults, description):
    """
    Execute a complete training run.
    """

    parameters = process_args(args, defaults, description)
   
    link_path = parameters.save_path + '/last_' + parameters.experiment_prefix

    nn_file = os.path.join(link_path, parameters.nn_file)
    handle = open(nn_file, 'r')
    network = cPickle.load(handle)
    
    client = pymongo.MongoClient(host = parameters.mongo_host,
                                 port = parameters.mongo_port)
                                 
    
    db = client[parameters.experiment_prefix] 

    dataset = MongoDataset(db,'training_data',
                         obs_shape=(defaults.RESIZED_WIDTH,
                                    defaults.RESIZED_HEIGHT),
                         act_shape = (1,),
                         hist_len = parameters.phi_length,
                         act_type='int32'
                        )
                        
    param_server = ParameterServer(db,'params')


    trainer = NetworkTrainer(network,
                            dataset,
                            param_server,
                            batch_size = parameters.batch_size,
                            update_freq = parameters.update_frequency,
                            min_samples = parameters.replay_start_size,
                            max_updates = 
                            parameters.steps_per_epoch*parameters.epochs
                            )
    

    trainer.do_training()

if __name__ == '__main__':
    pass
