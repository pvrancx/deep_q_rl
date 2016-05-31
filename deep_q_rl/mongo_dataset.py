# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:33:02 2016

Copyright Peter Vrancx
"""

import numpy as np
import cPickle as pickle
import pymongo
from bson.objectid import ObjectId
from bson import Binary
import copy

class MongoDataset(object):
    def __init__(self,db,exp_name, obs_shape, act_shape = (1,), hist_len=4, 
                 obs_type = np.float, act_type = np.float):
        self._db = db
        self._collection = self._db[exp_name]
        self._collection.remove({})
        self.hist_len = hist_len
        self.act_shape = act_shape
        self.obs_shape = obs_shape
        self.obs_type = obs_type
        self.act_type = act_type
        
    def add_sample(self,obs, action, reward, terminal,
                   ep_id=0,step_id=0,agent_id=0):
        trans = {
            'obs': Binary( pickle.dumps( copy.copy(obs), 
                                                       protocol=2) ),
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'agent': agent_id,
            'ep_id':ep_id,
            'step_id':step_id,
            'r_idx': np.random.rand()
        
        
        }
        self._collection.insert_one(trans)
        
    def random_batch(self, batch_size):
        count = 0
        assert batch_size < self._collection.count(), 'too few samples in db'
        while count< batch_size:
            query = {"r_idx": {"$lte": np.random.rand()},"terminal": False}
            result = self._collection.find(query).sort('r_idx').limit(batch_size)
            count = result.count()
            
        phis = np.zeros((batch_size,)+self.obs_shape+(self.hist_len,),
                       dtype= self.obs_type)
        phis_next = np.zeros_like(phis)
        acts = np.zeros((batch_size,)+self.act_shape, dtype = self.act_type)
        rewards = np.zeros(batch_size)
        term = np.zeros(batch_size, dtype=bool)
        
        
        for batch_idx, r in enumerate(result):
            #steps needed to make transition
            #phis is obs[i:i+hist_len], phi_next is obs[i+1,i+hist_len+1]
            steps = range(r['step_id']-self.hist_len+1,r['step_id']+2)
            query = {'ep_id': r['ep_id'],'step_id': {'$in': steps}}
            trans = self._collection.find(query).sort('step_id',-1)
            #go through steps in descending order
            for t_idx, st in enumerate(trans):
                #epsiode end, stop history
                if t_idx > 0 and st['terminal']:
                    break
                obs = pickle.loads(st['obs'])
                
                #phis is obs[i:i+hist_len], phi_next is obs[i+1,i+hist_len+1]
                if t_idx != 0: #last obs is not part of phi
                    phis[batch_idx,:,-t_idx] = copy.copy(obs)
                if t_idx < self.hist_len:   
                    phis_next[batch_idx,:,-(t_idx+1)] = copy.copy(obs)
                
                #check last step to see if phi_next is terminal
                if t_idx == 0:
                    term[batch_idx] = st['terminal']
                
                #this is last obs of phi (next to last one of phi_next)
                #get action and reward for transition
                if t_idx == 1:
                    acts[batch_idx,] = st['action']
                    rewards[batch_idx] = st['reward']
                    
            self._collection.update_one({
                '_id': ObjectId(r['_id'])
                },{
                '$set': {
                'r_idx': np.random.rand()
                }
                }, upsert=False)
                    
        
        return (phis,acts,rewards,term,phis_next)
                       
            
        