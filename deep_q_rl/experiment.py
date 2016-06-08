"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import time
import numpy as np

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class GymExperiment(object):
    def __init__(self, env, agent, preprocessor, num_epochs, epoch_length, 
                 test_length, rng, progress_frequency):
        self.env = env
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.preprocessor = preprocessor


        self.rng = rng
        self.progress_frequency = progress_frequency
        self.experiment_start_time = time.time()


    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)

            total_epoch_time = time.time() - epoch_start_time
            average_epoch_time = (self.epoch_length+self.test_length)/total_epoch_time
            logging.info("Finished training + testing epoch {}, took {:.2f}s for {}+{} steps".format(
                         epoch, total_epoch_time, self.epoch_length, self.test_length) +
                         " ({:.2f} steps/s on avg)".format(average_epoch_time))
            logging.info("Expecting the experiment ({} epochs ) to take about {:.2f} seconds longer".format(
                         self.num_epochs - epoch, (self.num_epochs - epoch) * average_epoch_time))

        logging.info("Finished experiment, took {}s".format(
                    time.time() - self.experiment_start_time))
        logging.shutdown()


    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined by
        the number of steps executed. An epoch will be cut short if not enough
        steps are left. Prints a progress report after every trial.

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training
        """
        self.steps_left_this_epoch = num_steps
        prefix = "testing" if testing else "training"
        logging.info("Starting {} epoch {}/{}".format(prefix, epoch,
            self.num_epochs))
        epoch_start_time = time.time()
        self.last_progress_time = epoch_start_time

        # It's less pretty, keeping track through self.steps_left_this_epoch,
        # but it's decidedly better for logging throughout the experiment
        while self.steps_left_this_epoch > 0:
            logging.debug(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(self.steps_left_this_epoch))
            _, steps_run = self.run_episode(self.steps_left_this_epoch, testing)

        total_time = time.time() - epoch_start_time
        logging.info("Finished {} epoch {}; took {:.2f} seconds for {} steps ({:.2f} steps/s on avg)".format(
                        prefix, epoch, total_time, num_steps, num_steps / total_time))


    def _get_obs(self,obs):
        if self.preprocessor is None:
            return obs
        else:
            return self.preprocessor.get_observation(obs)


    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        obs = self.env.reset()

        action = self.agent.start_episode(self._get_obs(obs))
        num_steps = 0
        while True:
            obs,reward,terminal,_ = self.env.step(action)

            num_steps += 1
            self.steps_left_this_epoch -= 1

            if self.steps_left_this_epoch % self.progress_frequency == 0:
                time_since_last = time.time() - self.last_progress_time
                logging.info("steps_left:\t{}\ttime spent on {} steps:\t{:.2f}s\tsteps/second:\t{:.2f}".format
                             (self.steps_left_this_epoch, self.progress_frequency, 
                              time_since_last, self.progress_frequency / time_since_last))
                self.agent.report()
                self.last_progress_time = time.time()

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self._get_obs(obs))
        return terminal, num_steps


  