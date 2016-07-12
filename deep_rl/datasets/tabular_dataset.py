"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import theano
import logging
from collections import deque
import copy

floatX = theano.config.floatX

class DataSetInterface(object):
    '''
    Minimum interface for datasets
    '''
    def clear(self):
        pass
    
    def add_sample(S,A,R,T,ep,st,ag):
        pass
    
    def __len__(self):
        return 0
        
    def get_batch(self,batch_size):
        return None
        
    def phi(self, obs):
        pass
        
class SequentialDataSet(DataSetInterface):
    '''
    Replay memory that stores and returns samples in FIFO order
    
    '''
    
    def __init__(self,max_steps, phi_length):
        ''' Create a dataset '''
    
        self._obs = deque()
        self._acts = deque()
        self._rews = deque()
        self._terms = deque()
        
        self.max_steps = max_steps
        self.phi_length = phi_length
        
        
    def _n_steps(self):
        return len(self._obs)
        
    def _n_term(self):
        return np.sum(self._terms)
        
    def add_sample(self,s,a,r,t,*args):
        self._obs.append(s)
        self._acts.append(a)
        self._rews.append(r)
        self._terms.append(t)
        
        if self._n_steps() > self.max_steps:
            self._obs.popleft()
            self._rews.popleft()
            self._acts.popleft()
            self._terms.popleft()
            
    def clear(self):
        self._obs.clear()
        self._acts.clear()
        self._terms.clear()
        self._rews.clear()
        
    def phi(self, obs):
        """Return a sequence of observations, using the last phi_length -
        1, plus obs.

        """

        phi = np.empty((self.phi_length,) +self._obs[-1].shape, dtype=floatX)
        #phi[0:self.phi_length - 1] = np.array(self._obs[-self.phi_length+1:])
        for i in xrange(-1,-self.phi_length,-1):
            phi[i-1] = self._obs[i]
        phi[-1] = obs
        return phi
        
        
    def __len__(self):
        return max(0,self._n_steps()- self.phi_length - 
            self._n_term() * self.phi_length)
            
    def _pop_step(self):
        s = self._obs.popleft()
        a = self._acts.popleft()
        r = self._rews.popleft()
        t = self._terms.popleft()
        
        return s,a,r,t
            
    def get_batch(self, batch_size,**kwargs):
        '''
        returns min(batch_size, len(self)) transition samples in FIFO order
    
        '''
        states = []
        rews = []
        next_states = []
        terms = []
        acts = []
        
        current_state = deque()
        
        while not self._n_steps()==0 and len(states) < batch_size:
            s,a,r,t = self._pop_step()

            if len(current_state) == self.phi_length:
                states.append(copy.deepcopy(current_state))
                rews.append(r)
                acts.append(a)
                current_state.popleft()
                current_state.append(s)
                terms.append(t)
                next_states.append(copy.deepcopy(current_state))
            else:
                current_state.append(s)
            
            #terminal states can't be transition starts
            if t:
                current_state.clear()
                
        return (np.array(states,dtype=floatX),
                np.array(acts,dtype='int32')[:,np.newaxis],
                np.array(rews,dtype=floatX)[:,np.newaxis],
                np.array(next_states,dtype=floatX),
                np.array(terms,dtype='int32')[:,np.newaxis]
                )
            
        
        

class DataSet(DataSetInterface):
    """A replay memory consisting of circular buffers for observed images,
    actions, and rewards.
    """
    def __init__(self, rng, obs_shape,obs_type=floatX,act_type='uint8',
                 max_steps=1000, phi_length=4):
        """Construct a DataSet.

        Arguments:
            
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches
        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.obs_shape = obs_shape
        self.obs_type = obs_type
        self.act_type = act_type
 
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.obs = np.zeros((max_steps,) + self.obs_shape, dtype=obs_type)
        self.actions = np.zeros(max_steps, dtype=act_type)
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0    # Points to free index
        self.size = 0
        
    def clear(self):
        self.obs = np.zeros((self.max_steps,) + self.obs_shape, 
                            dtype=self.obs_type)
        self.actions = np.zeros(self.max_steps, dtype=self.act_type)
        self.rewards = np.zeros(self.max_steps, dtype=floatX)
        self.terminal = np.zeros(self.max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0    # Points to free index
        self.size = 0

    def add_sample(self, img, action, reward, terminal,episode,step,agent):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.obs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.obs.take(indexes, axis=0, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length,) +self.obs_shape, dtype=floatX)
        phi[0:self.phi_length - 1] = self.obs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    def get_batch(self, batch_size,random=True):
        """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size  state transitions. If random = True transitions are
chosen randomely, otherwise last batch_size transitions are used

        """
        if batch_size > self.size:
            logging.debug('batchsize requested but size is '
                          +str(batch_size) +' '+str(self.size))
            return None
            
        # Allocate the response.
        states = np.zeros((batch_size,
                           self.phi_length)+self.obs_shape,
                           dtype=self.obs_type)
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')
        next_states = np.zeros((batch_size,
                                self.phi_length)+self.obs_shape,
                               dtype=self.obs_type)

        count = 0
        index = (self.bottom + self.size - self.phi_length-1)
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            if random:
                index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            initial_indices = np.arange(index, index + self.phi_length)
            transition_indices = initial_indices + 1
            end_index = index + self.phi_length - 1
            
            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                index -= 1
                continue

            # Add the state transition to the response.
            states[count] = self.obs.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.obs.take(transition_indices,
                                                axis=0,
                                                mode='wrap')
            count += 1
            index -=1

        return states, actions, rewards, next_states, terminal


