"""
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


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
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

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_out = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)
                                        
        #last hidden layer
        self.l_hid = lasagne.layers.get_all_layers(self.l_out)[-2]
        self.num_phi = lasagne.layers.get_output_shape(self.l_hid)[-1]
        
                        
        # theano.compile.function_dump('network.dump', self.l_out)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))
                               
        #model
        phi = lasagne.layers.get_output(self.l_hid, states / input_scale) 
        next_phi = lasagne.layers.get_output(self.l_hid, 
                                             next_states / input_scale)
                        
        #do not backprop through features
        phi = theano.gradient.disconnected_grad(phi)
        next_phi = theano.gradient.disconnected_grad(next_phi)
        
        #linear model
        self.l_phi = lasagne.layers.DenseLayer(
            self.l_hid,
            num_units=self.num_actions * self.num_phi,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=lasagne.init.Constant(0.0)
        )
        self.l_rew = lasagne.layers.DenseLayer(
            self.l_hid,
            num_units=self.num_actions,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=lasagne.init.Constant(0.0)
        )
        pred_phi, pred_r = lasagne.layers.get_output([self.l_phi,self.l_rew],
                                                     states / input_scale)
        loss_r = T.mean(0.5*(rewards - pred_r[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1)))**2)
        
        
        
        indexes = T.arange(self.batch_size).dimshuffle('x', 0)
        #max index
        mask = T.lt(indexes,self.num_phi*(actions.dimshuffle(0, 'x')+1))
        #min index
        mask = (mask & T.ge(indexes, self.num_phi*(actions.dimshuffle(0, 'x'))))
        
        loss_phi = T.mean(0.5*mask*(pred_phi - 
                    T.tile(next_phi,[1,self.num_actions]))**2)

        #rew cost
        #/model

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

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)  
        #only train output layers for model
        params_r = [self.l_rew.W, self.l_rew.b] 
        params_phi = [self.l_phi.W, self.l_phi.b]
        
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        
        givens_phi = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            actions: self.actions_shared
        }
        
        givens_r = {
            states: self.states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared
        }
        
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)
                                                     
        updates_r = lasagne.updates.sgd(loss_r, params_r, self.lr)
        updates_phi = lasagne.updates.sgd(loss_phi, params_phi, self.lr)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._train_r = theano.function([], [loss_r,pred_r], updates=updates_r,
                                      givens=givens_r)
        self._train_phi = theano.function([], [loss_phi,pred_phi], 
                                          updates=updates_phi,
                                      givens=givens_phi)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})
        self._pred_r = theano.function([], pred_r,
                                       givens={states: self.states_shared})
        self._pred_phi = theano.function([], pred_phi,
                                       givens={states: self.states_shared})

    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if (network_type.endswith('cuda') or network_type.endswith('cudnn')) and \
                not theano.config.device.startswith("gpu"):
            prefix = re.sub(r'(_cudnn)|(_cuda)', '', network_type)
            cpu_version = "{}_cpu".format(prefix)
            logging.warn(network_type + " requested but no GPU found " +
                         "(device {}) ".format(theano.config.device) +
                         "defaulting to {}".format(cpu_version))
            network_type = cpu_version

        if network_type == "nature_cuda":
            return self.build_nature_network_cuda(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        # Requires cuDNN which is not freely available. 
        if network_type == "nature_cudnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        if network_type == "nature_cpu":
            return self.build_nature_network_cpu(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        if network_type == "nips_cuda":
            return self.build_nips_network_cuda(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        # Requires cuDNN which is not freely available. 
        if network_type == "nips_cudnn":
            return self.build_nips_network_dnn(input_width, input_height,
                                               output_dim, num_frames,
                                               batch_size)
        if network_type == "nips_cpu":
            return self.build_nips_network_cpu(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        if network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        if network_type == "attempt1_cpu":
            return self.build_attempt1_cpu(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        loss_r, _ = self._train_r()
        loss_phi, _ = self._train_phi()
        self.update_counter += 1
        return np.sqrt(loss),np.sqrt(loss_r),np.sqrt(loss_phi)

    def q_vals(self, state):
        # Might be a slightly cheaper way by reshaping the passed-in state,
        # though that would destroy the original
        states = np.zeros((1, self.num_frames, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        return self._q_vals()[0]

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_nature_network_cuda(self, input_width, input_height, output_dim,
                                  num_frames, batch_size):
        from lasagne.layers import cuda_convnet
        conv_layer = cuda_convnet.Conv2DCCLayer
        return self.build_nature_network(input_width, input_height, output_dim,
                             num_frames, batch_size, conv_layer)

    def build_nature_network_dnn(self, input_width, input_height, output_dim,
                                  num_frames, batch_size):
        from lasagne.layers import dnn
        conv_layer = dnn.Conv2DDNNLayer
        return self.build_nature_network(input_width, input_height, output_dim,
                                    num_frames, batch_size, conv_layer)

    def build_nature_network_cpu(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        from lasagne.layers import conv 
        conv_layer = conv.Conv2DLayer
        return self.build_nature_network(input_width, input_height, output_dim,
                                    num_frames, batch_size, conv_layer)

    def build_nature_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size, conv_layer):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = conv_layer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_nips_network_cuda(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        from lasagne.layers import cuda_convnet
        conv_layer = cuda_convnet.Conv2DCCLayer
        return self.build_nips_network(input_width, input_height, output_dim,
                                  num_frames, batch_size, conv_layer)

    def build_nips_network_dnn(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        # Import it here, in case it isn't installed.
        from lasagne.layers import dnn
        conv_layer = dnn.Conv2DDNNLayer
        return self.build_nips_network(input_width, input_height, output_dim,
                                  num_frames, batch_size, conv_layer)

    def build_nips_network_cpu(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        from lasagne.layers import conv
        conv_layer = conv.Conv2DLayer
        return self.build_nips_network(input_width, input_height, output_dim,
                                  num_frames, batch_size, conv_layer)

    def build_nips_network(self, input_width, input_height, output_dim,
                           num_frames, batch_size, conv_layer):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_attempt1_cpu(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        from lasagne.layers import conv
        conv_layer = conv.Conv2DLayer
        return self.build_attempt1(input_width, input_height, output_dim,
                                  num_frames, batch_size, conv_layer)

    def build_attempt1(self, input_width, input_height, output_dim,
                           num_frames, batch_size, conv_layer):
        l_in = lasagne.layers.InputLayer(
            # Batch size is undefined so we can chuck in as many as we please
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        # l_hidden1 = lasagne.layers.DenseLayer(
        #     l_conv2,
        #     num_units=256,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     #W=lasagne.init.HeUniform(),
        #     W=lasagne.init.Normal(.01),
        #     b=lasagne.init.Constant(.1)
        # )

        default_gate = lasagne.layers.Gate(
            W_in=lasagne.init.Normal(.01),
            W_hid=lasagne.init.Normal(.01),
            W_cell=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            nonlinearity=lasagne.nonlinearities.sigmoid
        )

        l_lstm1 = lasagne.layers.LSTMLayer(
                l_conv2,
                num_units=256,
                ingate = default_gate,
                outgate = default_gate,
                forgetgate = default_gate,
                cell = default_gate,
                # https://github.com/Lasagne/Lasagne/blob/290379e01e74c842aab5b91d4d46a6a207245684/lasagne/layers/recurrent.py#L1033
                nonlinearity=lasagne.nonlinearities.rectify,
                backwards=False, #default
                learn_init=True,
                peepholes=True, # Internal connection from cell to gates
                gradient_steps=100, # -1 is entire history 
                grad_clipping=1, # From alex graves' paper, not sure here
                precompute_input=True, # Should be a speedup
                only_return_final=True # Only need output for last frame
                )

        l_out = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_linear_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')


if __name__ == '__main__':
    main()
