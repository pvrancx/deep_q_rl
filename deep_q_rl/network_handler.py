# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:50:32 2016

@author: pvrancx
"""
import logging
import multiprocessing as mp
from multiprocessing.managers import SyncManager

class NetworkHandler(object):
    
    def __init__(self,network,dataset=[],batch_size = 32):
        self._network = network
        self._dataset = dataset
        self.batch_counter = 0
        self.batch_size = batch_size
    
    def choose_action(self,phi,eps):
        return self._network.choose_action(phi,eps)
        
    def q_vals(self,states):
        return self._network.q_vals(states)
    
    '''
    Samples data and performs 1 training step of the local network.
    '''
    def train(self):
        S,A,R,Sp,T = self._dataset.get_batch(self.batch_size,random=True)
        loss = self._network.train(S,A,R,Sp,T)
        self.batch_counter += 1
        return loss
    
    
class RemoteNetworkHandler(NetworkHandler):
    def __init__(self,network,host='localhost',port=50000,
                 authkey='',**kwargs):
        super(RemoteNetworkHandler,self).__init__(network,**kwargs)
        SyncManager.register('get_space')
        logging.debug('connecting to parameter server')
        self.param_server = SyncManager(address=(host,port),authkey=authkey)
        self.param_server.connect()
        self._global_space = self.param_server.get_space()

        
    '''
    Simply retrieves latest network network from server.
    Assumes actual updating is handeled by other process
    '''
    def train(self):
        params,n_updates,loss = self._global_space.get('params')
        self._network.set_params(params)
        self.batch_counter = n_updates
        return loss
        
        
class AsyncNetworkHandler(RemoteNetworkHandler):

       
    '''
    Retrieves global shared parameters and performs single
    training step on them. Other processes may update these
    global parameters as well.
    '''
    def train(self):
        if len(self._dataset) < self.batch_size:
            pass
        #apply update to global params, using local data
        super(AsyncNetworkHandler,self).train()
        S,A,R,Sp,T = self._dataset.get_batch(self.batch_size,random=False)
        loss = self._network.train(S,A,R,Sp,T)
        self.batch_counter += 1
        #this may overwrite updates by other processes
        self._global_space.update({'params':
                                    (self._network.get_params(),
                                     self.batch_counter,
                                     loss),
                                     'process':mp.current_process().name
                                     })
        self._dataset.clear()
        return loss