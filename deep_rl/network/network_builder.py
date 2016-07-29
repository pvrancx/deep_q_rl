# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:25:47 2016

@author: pvrancx
"""

import lasagne

class NetworkBuilder(object):
    @staticmethod
    def create_input_layer(input_shape,input_var = None, batch_size = None):
        return lasagne.layers.InputLayer(
                    shape=(batch_size,)+ input_shape,
                    input_var = input_var
                )
                
    @staticmethod            
    def add_dense_layer(net,size,
                        nonlinearity=lasagne.nonlinearities.rectify):
                            
        return lasagne.layers.DenseLayer(
                net,
                num_units=size,
                nonlinearity=nonlinearity,
                W=lasagne.init.Normal(.01),
                b=lasagne.init.Constant(.1)
                )
                
    @staticmethod
    def add_conv2D_layer(net,n_filters,
                       filter_size,
                       stride=(1,1),
                       nonlinearity=lasagne.nonlinearities.rectify,
                        conv_type='cpu'):
        if conv_type == 'cpu':
            conv_layer = lasagne.layers.Conv2DLayer
        elif conv_type == 'dnn':
            conv_layer = lasagne.layers.dnn.Conv2DDNNLayer
        elif conv_type == 'gemm':
            conv_layer = lasagne.layers.corrmm.Conv2DMMLayer
        elif conv_type == 'cuda':
            conv_layer = lasagne.layers.cuda_convnet.Conv2DCCLayer
        else:
            raise RuntimeError('unknown convolution type')
        
        return  conv_layer(
                    net,
                    num_filters=n_filters,
                    filter_size=filter_size,
                    stride=stride,
                    nonlinearity=nonlinearity,
                    W=lasagne.init.Normal(.01),
                    b=lasagne.init.Constant(.1)
                    )
                    
    @staticmethod
    def build_nips(input_shape,output_dim,batch_size=None,conv_type='cpu',**kwargs):
        net = NetworkBuilder.create_input_layer(input_shape,
                                                batch_size=batch_size)
        net = NetworkBuilder.add_conv2D_layer(
                net,
                n_filters=16,
                filter_size=(8, 8),
                stride=(4, 4),
                conv_type=conv_type)

        net = NetworkBuilder.add_conv2D_layer(
                net,
                n_filters=32,
                filter_size=(4, 4),
                stride=(2, 2),
                conv_type=conv_type)

        net = NetworkBuilder.add_dense_layer(
                net,
                size=256)

        net = NetworkBuilder.add_dense_layer(
                net,
                size=output_dim,
                nonlinearity=None)
        return net
     
    @staticmethod
    def build_linear(input_shape,output_dim,batch_size=None,**kwargs):
        net = NetworkBuilder.create_input_layer(input_shape,
                                                batch_size=batch_size)
        net = NetworkBuilder.add_dense_layer(
                net,
                size=output_dim,
                nonlinearity=None)
        return net
    
    @staticmethod    
    def build_mlp(input_shape,
                  output_dim,
                  sizes=[20],
                  batch_size = None,
                  nonlinearity=lasagne.nonlinearities.rectify,**kwargs):
        net = NetworkBuilder.create_input_layer(input_shape,
                                                batch_size=batch_size)
        for s in sizes:
            net = NetworkBuilder.add_dense_layer(
                    net,
                    size=s,
                    nonlinearity=nonlinearity)
        net = NetworkBuilder.add_dense_layer(
                net,
                size=output_dim,
                nonlinearity=None)
        return net
        
    @staticmethod
    def build_nature(input_shape,output_dim,batch_size= None,conv_type= 'cpu',
                     **kwargs):
        net = NetworkBuilder.create_input_layer(input_shape,
                                                batch_size=batch_size)
        net = NetworkBuilder.add_conv2D_layer(
                net,
                n_filters=32,
                filter_size=(8, 8),
                stride=(4, 4),
                conv_type=conv_type)

        net = NetworkBuilder.add_conv2D_layer(
                net,
                n_filters=64,
                filter_size=(4, 4),
                stride=(2, 2),
                conv_type=conv_type)
                
        net = NetworkBuilder.add_conv2D_layer(
                net,
                n_filters=64,
                filter_size=(3, 3),
                stride=(1, 1),
                conv_type=conv_type)
                
        net = NetworkBuilder.add_dense_layer(
                net,
                size=512)

        net = NetworkBuilder.add_dense_layer(
                net,
                size=output_dim,
                nonlinearity=None)

        return net
        
    @staticmethod
    def build_network(network_type,input_shape,output_dim,batch_size=None,
                      **kwargs):
        if network_type == 'mlp':
            build_net = NetworkBuilder.build_mlp
        elif network_type == 'linear':
            build_net = NetworkBuilder.build_linear
        elif network_type == 'nips':
            build_net = NetworkBuilder.build_nips
        elif network_type == 'nature':
            build_net = NetworkBuilder.build_nature
        else:
            raise RuntimeError('unknown network type')
        return build_net(input_shape=input_shape,
                         output_dim = output_dim,
                         batch_size = batch_size,
                         **kwargs)
        
        
        