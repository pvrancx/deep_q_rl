#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

"""

import network_launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 50000
    EPOCHS = 100
    ROM = 'breakout.bin'


    # ----------------------
    # Agent/Network parameters:
    
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 10            # upload new parameters every 50 steps
    BATCH_SIZE = 32
    REPLAY_START_SIZE = 1000
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84

if __name__ == "__main__":
    network_launcher.launch(sys.argv[1:], Defaults, __doc__)
