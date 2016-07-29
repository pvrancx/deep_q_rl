"""
pvrancx

Based on:

Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague


"""
import re
import logging
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop
from network_builder import NetworkBuilder

class DeepLearner(object):
    def __init__(self, input_shape, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 conv_type = 'cpu'):

        #store settings
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.batch_accumulator = batch_accumulator
        self.input_scale = input_scale
        self.conv_type = conv_type
        
        lasagne.random.set_rng(self.rng)

        self.update_counter = 0
        
        #create data buffers
        data_shape = (batch_size, num_frames)+ input_shape
        self._create_buffers(data_shape)
        
        #placeholder variables
        in_shape = (num_frames,)+input_shape
        self.states = T.TensorType(theano.config.floatX, 
                             [False] * (len(in_shape)+1))(name ='states')
        self.next_states = T.TensorType(theano.config.floatX, 
                             [False] * (len(in_shape)+1))(name='next_states')
        self.rewards = T.col('rewards')
        self.actions = T.icol('actions')
        self.terminals = T.icol('terminals')
            
        #setup network training
        self.l_out = None # should be set by _build_network
        self._build_network(network_type)
        self._loss = self._get_loss()
        self._params = self._get_all_params()
        self._givens = self._get_givens()
        self._updates = self._get_updates(self._loss,self._params,update_rule)
        
        #internal training function
        self._train = theano.function([], 
                                      [self._loss ], 
                                      updates=self._updates,
                                      givens=self._givens)
                                      
        #gradients for external training                
        grads = T.grad(self._loss,self._params)
        self._grads = theano.function([], 
                                      grads+[self._loss],givens=self._givens)
                                      
    #subclasses should implement these   
    def _get_loss(self):
        pass
    
    def _get_all_params(self):
        pass
    
    def _build_network(self, network_type):
        in_shape = (self.num_frames,)+self.input_shape
        self.l_out = NetworkBuilder.build_network(network_type, in_shape,
                                        self.num_actions, self.batch_size,
                                        conv_type = self.conv_type 
                                        )
        # theano.compile.function_dump('network.dump', self.l_out)
        if self.freeze_interval > 0:
            self.next_l_out = NetworkBuilder.build_network(network_type, 
                                        in_shape,
                                        self.num_actions, self.batch_size,
                                        conv_type = self.conv_type 
                                        )
            self.reset_target()
    
    def _get_givens(self):
        givens = {
            self.states: self.states_shared,
            self.next_states: self.next_states_shared,
            self.rewards: self.rewards_shared,
            self.actions: self.actions_shared,
            self.terminals: self.terminals_shared
        }
        return givens
                                      
        
    def _get_updates(self,loss,params,update_rule):
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        elif update_rule == 'adadelta':
            updates = lasagne.updates.adadelta(loss, params)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)
                                                     
        return updates
        
    def _create_buffers(self,data_shape):
        self.states_shared = theano.shared(
            np.zeros(data_shape,
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros(data_shape,
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((data_shape[0], 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((data_shape[0], 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((data_shape[0], 1), dtype='int32'),
            broadcastable=(False, True))
        
    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train network on a single batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self._load_data(states, actions, rewards, next_states, terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_target()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)
        
    
    def grads(self, states, actions, rewards, next_states, terminals):
        """
        returns gradients for one batch. Does not update parameters.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: gradients, average loss
        """

        self._load_data(states, actions, rewards, next_states, terminals)
        result = self._grads()
        grads, loss = result[:-1],result[-1]
        return (grads,np.sqrt(loss))
    
    def _load_data(self, states, actions, rewards, next_states, terminals):
        '''loads batch in shared memory buffers'''
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        
    def set_params(self,params):
        '''sets network parameters to provided values'''
        lasagne.layers.helper.set_all_param_values(self.l_out, params)

    def get_params(self):
        '''Returns current network parameter values'''
        return lasagne.layers.helper.get_all_param_values(self.l_out)
        
    def get_target_params(self):
        '''Returns current target network parameter values'''
        return lasagne.layers.helper.get_all_param_values(self.next_l_out)
         
        
    def set_target_params(self,params):
        '''sets target parameters to provided parameter values'''
        lasagne.layers.helper.set_all_param_values(self.next_l_out, params)

    def reset_target(self):
        '''sets target parameter values to current network parameter values'''
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)
        
    def choose_action(self,state,*args):
        '''random action selection'''
        return np.random.randint(self.num_actions)


class DeepQLearner(DeepLearner):
    """
    Deep Q-learning network
    """

    def __init__(self, input_shape, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 conv_type = 'cpu'):
        super(DeepQLearner,self).__init__(input_shape, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 conv_type = 'cpu')
        

        #create q_value function
        q_vals = lasagne.layers.get_output(self.l_out, 
                                           self.states/self.input_scale)
        
        
        self._q_vals = theano.function([], q_vals,
                                       givens={self.states: self.states_shared})
                                       
                            
    def _get_loss(self):
        '''DQN loss'''
        #network output
        q_vals = lasagne.layers.get_output(self.l_out, 
                                           self.states/self.input_scale)
         #target network
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    self.next_states / self.input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    self.next_states / self.input_scale)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (self.rewards +
                  (T.ones_like(self.terminals) - self.terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(self.batch_size),
                               self.actions.reshape((-1,))].reshape((-1, 1))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if self.batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif self.batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(self.batch_accumulator))
            
        return loss

    def _get_all_params(self):
        return lasagne.layers.helper.get_all_params(self.l_out)  
  

    def q_vals(self, state):
        '''Returns all q-values for given state '''
        
        # Might be a slightly cheaper way by reshaping the passed-in state,
        # though that would destroy the original
        states = np.zeros((1, self.num_frames)+self.input_shape, 
                          dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        return self._q_vals()[0]
        
    def get_value(self,state):
        qvals = self.q_vals(state)
        return np.max(qvals)
        
    def choose_action(self, state, epsilon):
        '''epsilon greedy action selection'''
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)
        
    
        
        
class PolicyGradientNetwork(DeepLearner):
    '''
    A3C network
    '''
    def __init__(self, input_shape, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 conv_type = 'cpu', beta = 1e-3):
               
        self.beta = beta

        #Add variable and buffer to store return values  
        self.returns = T.col('returns')

        
        self.returns_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
            
        super(PolicyGradientNetwork,self).__init__(input_shape, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 conv_type = 'cpu')
     
        #create function to get state value and action probabilities
        vals,act_prob = lasagne.layers.get_output(
                            [self.l_out,
                             self.l_out_pol], 
                             self.states/self.input_scale)
        
        
        self._outp = theano.function([], [vals,act_prob],
                                       givens={self.states: 
                                       self.states_shared})
        
    def _build_network(self,network_type):
        #create single output value network
        in_shape = (self.num_frames,)+self.input_shape
        self.l_out = NetworkBuilder.build_network(network_type, in_shape,
                                        1, None,
                                        conv_type = self.conv_type 
                                        )
        # theano.compile.function_dump('network.dump', self.l_out)
        if self.freeze_interval > 0:
            self.next_l_out = NetworkBuilder.build_network(network_type, 
                                        in_shape,
                                        1, None,
                                        conv_type = self.conv_type 
                                        )
            self.reset_target()
        #remove output layer
        l_hid = lasagne.layers.helper.get_all_layers(self.l_out)[-2]
        #add action prob outputs, share lower layers
        self.l_out_pol = lasagne.layers.DenseLayer(l_hid,
                                num_units=self.num_actions,
                                nonlinearity=lasagne.nonlinearities.softmax,
                                W=lasagne.init.Normal(.01),
                                b=lasagne.init.Constant(.1)                                                   
                                )
        
    def _get_loss(self):
        '''A3C loss - combines policy gradient and value squared error'''
        #network outputs
        vals, act_probs =  lasagne.layers.get_output([self.l_out ,
                                                      self.l_out_pol], 
                                                      self.states /self.input_scale)
        #target values for policy advantage function
        #if self.freeze_interval > 0:
        #    target_vals = lasagne.layers.get_output(self.next_l_out,
        #                                            self.states/self.input_scale)
        #else:
        #    target_vals = lasagne.layers.get_output(self.l_out,
        #                                            self.states /self.input_scale)
        target_vals = theano.gradient.disconnected_grad(vals)
        
        H = - T.sum(act_probs * T.log(act_probs),axis =1 )
        
        #policy gradient loss
        loss = T.log(act_probs)[T.arange(self.actions.shape[0]),
                               self.actions.reshape((-1,))].reshape((-1, 1))\
                    *(self.returns - target_vals)\
         #           (1-self.terminals[-1,0])*target_vals[-1,0]))
        #add entropy regulizer
        #loss += self.beta * H
                   
        #add value approximation loss ->  assumes shared network structure
        loss += 0.5*(self.returns - vals)**2
        return T.sum(loss)
                                       
    def _get_all_params(self):
        return lasagne.layers.get_all_params([self.l_out ,
                                       self.l_out_pol])
    def _get_givens(self):
        givens = {
            self.states: self.states_shared,
            self.returns: self.returns_shared,
            self.actions: self.actions_shared,
           # self.terminals: self.terminals_shared
        }

        return givens
                                       
    def _load_data(self, states, actions, rewards, next_states, terminals):
        #get value of last state
        # IMPORTANT: do this first - this overwrites the states shared var
        last_value = self.get_value(next_states[-1])

        #load standard batch
        super(PolicyGradientNetwork, self)._load_data(states, 
                                                    actions, 
                                                    rewards, 
                                                    next_states, 
                                                    terminals)
        #also load returns
        returns = self.disc_rewards(rewards,terminals,self.discount,last_value)
        self.returns_shared.set_value(returns)        
                    
    def disc_rewards(self,rews,terms,gamma, last_val = 0.):
        '''calculate discounted returns given reward sequence'''
        #bootstrap if last ep is cut off
        R = 0. if terms[-1] else last_val
        result = np.zeros_like(rews,dtype=theano.config.floatX)
        #go through reward sequence backwards
        for i in reversed(xrange(0,rews.size)):
            #reset return at for new episodes
            if terms[i] and i < rews.size-1: R = 0.
            #accumulate discounted return
            R = rews[i] + gamma * R
            result[i] = R
        #normalize returns - reduces variance 
        std = np.std(result)
        #check for  - some settings give constant rewards
        if np.isnan(std) or std == 0.: std = 1.
        std_R = (result - np.mean(result))/std
        return std_R
        
  
    def set_params(self,params):
        '''sets network parameters to provided values'''
        lasagne.layers.helper.set_all_param_values([self.l_out,
                                                    self.l_out_pol],
                                                    params)

    def get_params(self):
        '''Returns current network parameter values'''
        return lasagne.layers.helper.get_all_param_values([self.l_out,
                                                           self.l_out_pol])
                                                           
    def get_value(self,state):
        value,_ = self.value_probs(state)
        return value[0]
                                                           
    def value_probs(self, state):
        '''Returns value and action probs for given state '''
        
        # Might be a slightly cheaper way by reshaping the passed-in state,
        # though that would destroy the original
        states = np.zeros((1, self.num_frames)+self.input_shape, 
                          dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        result = self._outp()   
        return (result[0][0],result[1][0])
        
    def choose_action(self, state,*args):
        '''on-policy action selection'''
        _,probs = self.value_probs(state)
        return np.random.choice(probs.size,p=probs)
    
