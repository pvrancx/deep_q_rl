# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:34:33 2016

@author: pvrancx
"""
import time
import logging

import multiprocessing as mp
from multiprocessing.managers import SyncManager


class NetworkTrainer(object):
    
    def __init__(self, 
                 network_manager,
                 min_samples = 50000,
                 max_updates = 5000000):
                     
      
        self.updates = 0
        self.min_samples = min_samples
        self.start_time = time.time()
        self.max_updates = max_updates
        
        self.manager = network_manager
        
    def do_training(self):
        self.start_time = time.time()
        while True:
            #check if we have enough samples in the db
            if len(self.manager._dataset) < self.min_samples:
                time.sleep(1) #wait for more samples
                logging.debug('waiting for training samples')
            else:
                self.manager.train()
                self.updates += 1
                if self.updates % 5000:
                    dt = time.time()-self.start_time
                    logging.info('training {:d}, total time for 5000 batches:\
                    {:.2f} s, steps/second: {:.2f}'.format(
                        self.updates,
                        dt,
                        dt/5000.)
                    )
                    self.start_time = time.time()
                
                if (self.max_updates != 0) and\
                    (self.updates >= self.max_updates):
                    break

            
                
                
        