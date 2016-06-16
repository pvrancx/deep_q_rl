# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:50:32 2016

@author: pvrancx
"""

class NetworkHandler(object):
    
    def __init__(self,network,dataset=None,batch_size = 32):
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
        S,A,R,Sp,T = self._dataset.random_batch(self.batch_size)
        loss = self._network.train(S,A,R,Sp,T)
        self.batch_counter += 1
        return loss
    
    
class RemoteNetworkHandler(NetworkHandler):
    def __init__(self,network,param_server,**kwargs):
        super.__init__(self,network,**kwargs) 
        self.param_server = param_server
        
    '''
    Simply retrieves latest network network from server.
    Assumes actual updating is handeled by other process
    '''
    def train(self):
        params,loss,n_updates =self.param_server.get_params()
        self._network.set_params(params)
        self.batch_counter = n_updates
        return loss
        
        
class AsyncNetworkHandler(NetworkHandler):
    def __init__(self,network,global_space,**kwargs):
        super.__init__(self,network,**kwargs) 
        self._global_space = global_space
       
    '''
    Retrieves global shared parameters and performs single
    training step on them. Other processes may update these
    parameters as well.
    '''
    def train(self):
        #apply update to global params, using local data
        self._network.set_params(self._global_space.params)
        loss = super.train(self)
        #this may overwrite updates by other processes
        self._global_space.params = self.network.get_params()
        return loss