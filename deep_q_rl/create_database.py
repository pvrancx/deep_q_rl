# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:51:53 2016

@author: pvrancx
"""

import pymongo

def create_db(exp_name,host='localhost',port=27017):
    client = pymongo.MongoClient(host, port)
    db = client[exp_name]
    p_collection = db['params']
    d_collection = db['training_data']
    d_collection.remove({})
    p_collection.remove({})
    d_collection.create_index([("ep_id", pymongo.ASCENDING),
                               ("step_id", pymongo.ASCENDING)], 
                            unique= True)
    d_collection.create_index("timestamp", expireAfterSeconds=20*60) 
    client.close()
    
    
if __name__ == '__main__':
    create_db('experiment0')