import theano

def configure_theano_for_profiling(save_path):
    theano.config.profile = True
    theano.config.profile_memory = True
    theano.config.profile_optimizer = False
    theano.config.profiling.debugprint = True
    theano.config.profiling.destination = '/'.join((save_path,
        'theano.profile'))
