# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:25:28 2016

Copyright Peter Vrancx
"""

import gym
import pymongo

from deep_q_rl.mongo_dataset import MongoDataset

env = gym.make('CartPole-v0')


client = pymongo.MongoClient()
db = client.test_database
sample_db = MongoDataset(db,'test_exp',
                         obs_shape=env.observation_space.high.shape,
                         act_shape = (env.action_space.n,))


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        sample_db.add_sample(observation,action,reward,done,i_episode,t)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break