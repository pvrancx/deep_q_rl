# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:34:33 2016

@author: pvrancx
"""
import time
import logging


class NetworkTrainer(object):
    
    def __init__(self,network,dataset, 
                 param_server,
                 batch_size = 32,
                 update_freq = 100, 
                 min_samples = 50000,
                 max_updates = 5000000):
                     
        self.dataset = dataset
        self.param_server = param_server
        self.network = network
        self.update_freq = update_freq
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.n_batches = 0
        self.param_server.add_params(self.network.get_params())
        self.start_time = time.time()
        self.max_updates = max_updates
        
    def do_training(self):
        self.start_time = time.time()
        while True:
            #check if we have enough samples in the db
            if len(self.dataset) < self.min_samples:
                time.sleep(1) #wait for more samples
                logging.debug('waiting for training samples')
            else:
                #sample minibatch
                S,A,R,Sp,T = self.dataset.random_batch(self.batch_size)
                loss = self.network.train(S,A,R,Sp,T)
                self.n_batches += 1
                
                #push params to db
                if self.n_batches % self.update_freq ==0:
                    logging.debug('pushing new params to db')
                    self.param_server.add_params(self.network.get_params(),
                                                 loss)
                                                 
                if self.n_batches % 5000:
                    dt = time.time()-self.start_time
                    logging.info('training {:d}, total time for 5000 batches:\
                    {:.2f} s, steps/second: {:.2f}'.format(
                        self.n_batches,
                        dt,
                        dt/5000.)
                    )
                    self.start_time = time.time()
                
                if (self.max_updates != 0) and\
                    (self.n_batches >= self.max_updates):
                    break

            
                
                
        