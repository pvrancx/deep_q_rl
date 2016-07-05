# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:50:32 2016

@author: pvrancx
"""
import logging
import math
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
        class ParamManager(SyncManager): pass
        ParamManager.register('get_space')
        logging.debug('connecting to parameter server')
        self.param_server = ParamManager(address=(host,port),authkey=authkey)
        self.param_server.connect()
        self._global_space = self.param_server.get_space()

        
    '''
    Retrieves latest network parameters from server.  
    
    Retrieves latest network parameters from server. Does not perform local
    training. Assumes actual parameter updating is handeled by other process
    '''
    def train(self):
        global_params,n_updates,loss = self._global_space.get_param_values()
        self._network.set_params(global_params)
        self.batch_counter = n_updates
        return loss
        
     
         
'''
Network handler that allows multiple processes to update the same parameter
values, without locking.
This implementation is actually partially syncrhonized since it relies on
a multiprocessing.Manager server and proxies to share values. This means
that read and write operations will be syncrhonized and cannot interfere
with those of other processes. 
Processes can still overwrite each others' updates, however.
'''        
         
class AsyncNetworkHandler(RemoteNetworkHandler):
    '''
    Asynchronous distributed network updating
    
    Network handler that allows multiple processes to update the same parameter
    values, without locking.
    This implementation is actually partially syncrhonized since it relies on
    a multiprocessing.Manager server and proxies to share values. This means
    that read and write operations will be syncrhonized and cannot interfere
    with those of other processes. 
    ''' 
    def __init__(self,network,
                  max_loss = 3,
                  param_update_freq = 1,
                  target_update_freq = 1000,
                  clear_samples=False,
                  params = None,
                  **kwargs):
        super(AsyncNetworkHandler,self).__init__(network,**kwargs)
        self.clear_samples = clear_samples
        
        self.mu_loss = 0.
        self.var_loss =0.
        self.max_loss = max_loss
        self.update_freq = param_update_freq
        self.target_update_freq = target_update_freq
        self.last_target_update = 0
        self.n_steps = 0
        self.n_rejected = 0
        
        #parameters to update
        self.params = params
        if self.params is None: 
            #update all params
            self.params = self._global_space.get_params()
        

    '''
    Calculate network parameter gradients and push to parameter server.\
    '''
    
    def train(self):
        if len(self._dataset) < self.batch_size:
            logging.debug('too few samples to train')
            return 0.
            
        #sync local params with global
        if self.n_steps % self.update_freq == 0:
            super(AsyncNetworkHandler,self).train()
                    
            #check if target has been updated
            (global_target, global_steps) = self._global_space.get_target_param_values()
            if global_steps > self.last_target_update:
                self._network.set_q_hat(global_target)
                self.last_target_update = global_steps

        #single update step
        S,A,R,Sp,T = self._dataset.get_batch(self.batch_size,random=True)
        grads,loss = self._network.grads(S,A,R,Sp,T)#doesn't update local params
        
        #reject update if loss is too high
        if self.n_steps < 2:
            self.mu_loss = loss
            sigma = 0.
        else:
            sigma = math.sqrt(self.var_loss/(self.n_steps-1.))
            
            
        if loss <= self.mu_loss + self.max_loss * sigma:
            #accept update, update statistics
            self.n_steps += 1
            err = loss-self.mu_loss
            self.mu_loss += err/float(self.n_steps)
            self.var_loss += (err*err)
        else:
            self.n_rejected +=1
            logging.debug('high loss, update rejected. '+ 
                            'accepted: {:d} rejected: {:d}'.format(
                            self.n_steps, self.n_rejected)
                        
                        )

        
        #push update to global space
        self._global_space.update(self.params, grads, self.batch_counter, loss)
            
        if self.clear_samples:
            self._dataset.clear()
        return loss
        
