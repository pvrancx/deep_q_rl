# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:30:40 2016

@author: pvrancx
"""


from bson import Binary

import cPickle as pickle

class ParameterServer(object):
    
    def __init__(self, db, collection_name = 'params'):
        self._db = db
        self._param_collection = self._db[collection_name]
        
        
    def add_params(self,params,loss=0.,updates=0.):
        self._param_collection.insert_one(
            {'params': Binary( pickle.dumps(params) ), 
            'loss': loss,
            'updates': updates
            }
        )
        
        
    def get_params(self):
        if self._param_collection.count() < 1:
            return None
        p = self._param_collection.find().sort('$natural', -1).limit(1).next()
        return (pickle.loads(p['params']),p['loss'],p['updates'])
        
