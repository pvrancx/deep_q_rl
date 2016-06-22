"""
The NeuralAgent class wraps a deep Q-network for training and testing
in the Arcade learning environment.

original author: Nathan Sprague

adapted by Peter vrancx

"""

#import os
import cPickle
import time
import logging
import uuid

#import pymongo
#from mongo_dataset import MongoDataset


import numpy as np

#import ale_data_set



import sys
sys.setrecursionlimit(10000)

class NeuralAgent(object):

    def __init__(self, training_dataset, test_dataset, net_handler, 
                 epsilon_start, epsilon_min,
                 epsilon_decay, replay_start_size, 
                 update_frequency, rng, save_path, profile):

        self.network = net_handler
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng
        self.save_path = save_path
        self.profile = profile




        self.exp_dir = save_path
        self.num_actions = self.network._network.num_actions
        logging.info("Creating data sets")

        self.dataset = training_dataset
        self.test_dataset = test_dataset

        self.phi_length = self.dataset.phi_length

        
        #MongoDataset(db,'training_data',
        #                 obs_shape=(self.image_width,
#                                    self.image_height),
#                         act_shape = (1,),
#                        hist_len = self.phi_length,
#                        act_type='int32'
#                        )


        # just needs to be big enough to create phi's
#        MongoDataset(db,'test_data',
#                         obs_shape=(self.image_width,
#                                    self.image_height),
#                         act_shape = (1,),
#                        hist_len = self.phi_length,
#                        act_type='int32'
#                        )
#        
       # self.param_server = ParameterServer(db,'params')
       # self.param_server.add_params(self.network.get_params())
                        
        logging.info("Finished creating data sets")
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.epoch_start_episode = 0 
        self.episode_counter = 0
        self.batch_counter = 0      # Tracks amount of batches trained

        self.holdout_data = None

        self.agent_id = str(uuid.uuid4())

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None
        

        


    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,num_updates,total_reward,reward_per_episode,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{}\n".format(epoch, num_episodes,
                                            self.network.batch_counter,
                                            self.total_reward,
                                            self.total_reward / float(num_episodes),
                                            holdout_sum)
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.episode_reward = 0
        self.episode_counter += 1

        
        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action


    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1

        #TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action = self._choose_action(self.test_dataset, .05,
                                         observation, np.clip(reward, -1, 1))

        #NOT TESTING---------------------------
        else:

            if len(self.dataset) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action = self._choose_action(self.dataset, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    #self.batch_counter += 1
                    self.loss_averages.append(loss)

            else: # Still gathering initial random data...
                action = self._choose_action(self.dataset, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))


        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose an action based on
        the current policy.
        """

        data_set.add_sample(self.last_img, 
                            self.last_action, 
                            reward, 
                            False,
                            self.episode_counter,
                            self.step_counter,
                            self.agent_id
                            )
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        loss = self.network.train()

        return loss


    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            # perform last training step
            loss = self._do_training()
            if loss:
                self.loss_averages.append(loss)
            # Store the latest sample.
            self.dataset.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True,
                                     self.episode_counter,
                                     self.step_counter,
                                     self.agent_id
                                     )

            logging.debug("steps/second: {:.2f}".format(\
                            self.step_counter/total_time))

            if self.network.batch_counter > 0:
                self._update_learning_file()
                logging.debug("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))

    def report(self):
        pass


    def finish_epoch(self, epoch):
        # Things get really nasty in profiling mode
        if self.profile:
            return
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network._network, net_file, -1)
        net_file.close()

    def start_testing(self):
        self.testing = True
        self.total_reward = 0        
        #self.episode_counter = 0



    def finish_testing(self, epoch):
        start_time = time.time()
        self.testing = False
        holdout_size = 3200



        # TODO check out holdout size in original code
        # Keep a random subset of transitions to evaluate performance over time
        #edit: use test set as holdout source
        if self.holdout_data is None and len(self.test_dataset) > holdout_size:
            self.holdout_data = self.dataset.get_batch(holdout_size)[0]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))

        self._update_results_file(epoch, 
                                  self.episode_counter - self.epoch_start_episode,
                                  holdout_sum / holdout_size)

        total_time = time.time() - start_time
        logging.info("Finishing up testing took {:.2f} seconds".format(total_time))
        self.epoch_start_episode = self.episode_counter



if __name__ == "__main__":
    pass
