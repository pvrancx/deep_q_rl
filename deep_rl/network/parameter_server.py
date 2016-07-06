# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:05:57 2016

@author: pvrancx
"""
import theano
import theano.tensor as T
import lasagne
from updates import deepmind_rmsprop
from collections import OrderedDict
import copy
import logging
import cPickle as pickle
import numpy as np

logger = logging.getLogger()


class ParameterServer(object):
    
    def __init__(self,params,update_rule,
                 learning_rate, 
                 rho,
                 rms_epsilon, 
                 momentum,
                 target_update_freq = 1000,
                 max_delay = 10):
        
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        
        self.params = params
        self.target_update_freq = target_update_freq
        self.max_delay = max_delay
        
        #create gradient variables for each of the parameters
        self._grad_vars = []
        self._update_fns = [] #OrderedDict()
        for i,p in enumerate(params):
            logging.debug('param.name: '+str(p.name) )
            logging.debug('param size: '+str(p.get_value().shape))
            g = T.TensorType(theano.config.floatX, 
                             [False] * p.ndim)()
            self._grad_vars.append(g)
        
            #get updates for vars
            if update_rule == 'deepmind_rmsprop':
                updates = deepmind_rmsprop([g], [p], self.lr, 
                                       self.rho,self.rms_epsilon)
            elif update_rule == 'rmsprop':
                updates = lasagne.updates.rmsprop([g], [p], self.lr, 
                                              self.rho,
                                              self.rms_epsilon)
            elif update_rule == 'sgd':
                updates = lasagne.updates.sgd([g], [p], self.lr)
            else:
                raise ValueError("Unrecognized update: {}".format(update_rule))
            
            if self.momentum > 0:
                updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)
                
            logger.debug('creating grad fn grad '+str(g)+' param '+str(p))
            self._update_fns.append(theano.function([g],p, updates=updates))
            
        self.n_updates = 0
        self.target_updates = 0
        self.loss = 0.

        self.target_param_values = copy.deepcopy(self.get_param_values()[0])
            
    def update(self,params,gradients,steps,loss):
        ''' update parameters using provided gradients'''

        assert len(params) == len(gradients), 'wrong number of gradients'
        if (steps - self.n_updates) > self.max_delay:
            return #stale gradient, skip update
        for i,p in enumerate(params):
            try:
                self._update_fns[i](gradients[i])
            #Fix me: for some reason this update fails sometimes in asynchronous mode
            except TypeError:
                l =copy.copy(locals())
                with open('dump'+str(np.random.rand())+'.pkl','wb') as f:
                    pickle.dump(l,f)
                return 0.
            
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            #note copy needed?
            self.target_param_values = copy.deepcopy(self.get_param_values()[0])
            self.target_updates += 1
            self.loss = loss
            
    def get_params(self):
        return self.params
        
        
    def get_param_values(self):
        '''returns current parameter values'''
        return ([p.get_value() for p in self.params],self.n_updates,self.loss)
        
    def get_target_param_values(self):
        '''returns current target parameter values'''
        return (self.target_param_values, self.target_updates)
        
        
        