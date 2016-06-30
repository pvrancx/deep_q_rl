- The network `nature_cpu` seems faster than `nips_cpu`, at least without
  replay. Don't see why so far, since nature has more and bigger convolutional
  layers. Both use the same libraries. Settings for this test was nature's.

- mine - nips_cpu - first round was 22 steps/s
- mine - nips_cuda - first round was 29.5 steps/s
- mine on cluster - nips_cpu first round was 73steps/s
- mine on cluster - nips_cuda first round was 68.55steps/s
- mine on cluster - nips_cudnn first round was 78.73steps/s

The _cpu version is actually sneaky and uses Theano underneath. Theano is free
to do as it pleases.

There's the possibility of using Theano's meta-optimizer which tries a bunch of
layers.
http://deeplearning.net/software/theano/library/tensor/nnet/conv.html

TODO: Profile. But not the optimizer, that one is broken.

