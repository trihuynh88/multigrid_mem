# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import time
import sys

from model import MNISTRecallModel
from constants import LOCAL_T_MAX
from constants import GRAD_NORM_CLIP
from constants import BATCH_SIZE
from constants import LOG_FILE
from constants import SAVE_MODEL_INTERVAL
from constants import SAVE_VIS_INTERVAL
from constants import CHECKPOINT_DIR
from constants import END_DECAY_TIME
from constants import END_ALPHA
from constants import INPUT_SIZE
from constants import LOG_INTERVAL

import pdb

import sys
import cv2
import os
import copy
from random import shuffle
from mnist import MNIST


class TrainingThread(object):
  def __init__(self,
	       network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):
	       
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    network.prepare_loss()

    with tf.device(device):
      var_refs = [v._ref() for v in network.get_vars()]
      self.gradients = tf.gradients(
        network.loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

      clipped_gradients, _  = tf.clip_by_global_norm(self.gradients, GRAD_NORM_CLIP)
      self.apply_gradients = grad_applier.apply_gradients(
      network.get_vars(),
      clipped_gradients)
     
    self.batch_size = BATCH_SIZE

    self.initial_learning_rate = initial_learning_rate

    # variable controling log output
    self.prev_global_t_log = 0
    self.prev_global_t_model = 0
    self.prev_global_t_vis = 0

    self.network = network

    # MNIST
    mndata = MNIST('mnist')
    # training set
    self.mnist_train_images_orig, self.mnist_train_labels_orig = mndata.load_training()
    self.mnist_train_images_orig = np.asarray(self.mnist_train_images_orig)
    self.mnist_train_labels_orig = np.asarray(self.mnist_train_labels_orig).reshape((-1,1))
    self.mnist_train_images_curr = self.mnist_train_images_orig
    self.mnist_train_labels_curr = self.mnist_train_labels_orig
    self.mnist_train_index_curr = 0
    # testing set
    self.mnist_test_images_orig, self.mnist_test_labels_orig = mndata.load_testing()
    self.mnist_test_images_orig = np.asarray(self.mnist_test_images_orig)
    self.mnist_test_labels_orig = np.asarray(self.mnist_test_labels_orig).reshape((-1,1))
    self.mnist_test_images_curr = self.mnist_test_images_orig
    self.mnist_test_labels_curr = self.mnist_test_labels_orig
    self.mnist_test_index_curr = 0
   
  def vis_save(self,sess,timestep):
    vis_dir = CHECKPOINT_DIR+("/%d"%timestep)
    if not os.path.exists(vis_dir):
      os.mkdir(vis_dir)
    num_steps = LOCAL_T_MAX
    for set_ind in range(5):
      [batch_input, batch_input2, batch_output, chosen_ind] = self.generate_test_input_output(num_steps)
      batch_input = batch_input.reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,1))
      batch_input2 = batch_input2.reshape((self.batch_size,INPUT_SIZE,INPUT_SIZE,1))
      batch_output = batch_output.reshape(self.batch_size) 
  
      cur_lstm_states = self.network.get_initial_state()
      list_to_run = [self.network.prediction]
      list_keys = self.network.lstm_state_ph_list + [self.network.input, self.network.input2, self.network.step_size]
      list_vals = cur_lstm_states + [batch_input, batch_input2, [num_steps]]
      list_output = sess.run( list_to_run,
              feed_dict = dict(zip(list_keys, list_vals)) )
      prediction_output = list_output[0]    
  
      batch_input = batch_input.reshape((num_steps,self.batch_size,INPUT_SIZE,INPUT_SIZE))
      batch_input2 = batch_input2.reshape((self.batch_size,INPUT_SIZE,INPUT_SIZE))
      for vis_ind in range(self.batch_size):
          vis_width = 28*num_steps + 5*(num_steps-1)
          vis_height = 28*2 + 5
          vis_img = np.ones((vis_height,vis_width))*(0.5)
          for step in range(num_steps):
              vis_img[:28, step*(28+5):step*(28+5)+28] = batch_input[step,vis_ind,:,:]
          vis_img[28+5:, chosen_ind*(28+5):chosen_ind*(28+5)+28] = batch_input2[vis_ind,:,:]
          scale_factor = 10
          vis_img_scaled = np.kron(vis_img, np.ones((scale_factor,scale_factor)))
          vis_img_scaled = (vis_img_scaled*255).astype(np.uint8)
          vis_img_scaled_3chan = np.stack((vis_img_scaled, vis_img_scaled, vis_img_scaled),axis=-1)
          cur_pred = prediction_output[vis_ind,:].argmax(-1)
          font                   = cv2.FONT_HERSHEY_SIMPLEX
          fontScale              = 10
          if (cur_pred == batch_output[vis_ind]):
            fontColor            = (0,255,0)
          else:
            fontColor            = (0,0,255)
          lineType               = 2
          loc_col = (chosen_ind+1)*(28+5)*scale_factor + 20
          loc_row = (28+5+28)*scale_factor - 20
          bottomLeftCornerOfText = (loc_col,loc_row)
          cv2.putText(vis_img_scaled_3chan, "%d"%(cur_pred), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
          ind_string = ("%02d_%02d"%(set_ind,vis_ind))
          save_path = vis_dir + "/" + ind_string + ".jpg"
          cv2.imwrite(save_path,vis_img_scaled_3chan)


  def _anneal_learning_rate(self, global_time_step):
    if global_time_step<END_DECAY_TIME:
      learning_rate = self.initial_learning_rate+(END_ALPHA-self.initial_learning_rate) * global_time_step / END_DECAY_TIME
    else:
      learning_rate = END_ALPHA
    return learning_rate

  def _record_params(self, sess, summary_writer, summary_op, loss_input, loss, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      loss_input: np.asscalar(loss)
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def generate_test_input_output(self, num_step):
    if (self.mnist_test_index_curr+self.batch_size*num_step)>len(self.mnist_test_labels_curr):
      per = np.random.permutation(len(self.mnist_test_labels_curr))
      self.mnist_test_images_curr = self.mnist_test_images_orig[per]
      self.mnist_test_labels_curr = self.mnist_test_labels_orig[per]
      self.mnist_test_index_curr = 0
    batch_input = self.mnist_test_images_curr[self.mnist_test_index_curr:self.mnist_test_index_curr+self.batch_size*num_step,:].reshape((num_step,self.batch_size,28,28,1))
    batch_input_label = self.mnist_test_labels_curr[self.mnist_test_index_curr:self.mnist_test_index_curr+self.batch_size*num_step,:].reshape((num_step,self.batch_size,1))
    chosen_ind = random.randint(0,num_step-2)
    batch_input2 = batch_input[chosen_ind,:,:,:,:] # query
    batch_output = batch_input_label[chosen_ind+1,:,:]
    self.mnist_test_index_curr += self.batch_size*num_step
    return [batch_input, batch_input2, batch_output, chosen_ind]    

  def generate_input_output(self, num_step):
    if (self.mnist_train_index_curr+self.batch_size*num_step)>len(self.mnist_train_labels_curr):
      per = np.random.permutation(len(self.mnist_train_labels_curr))
      self.mnist_train_images_curr = self.mnist_train_images_orig[per]
      self.mnist_train_labels_curr = self.mnist_train_labels_orig[per]
      self.mnist_train_index_curr = 0
    batch_input = self.mnist_train_images_curr[self.mnist_train_index_curr:self.mnist_train_index_curr+self.batch_size*num_step,:].reshape((num_step,self.batch_size,28,28,1))
    batch_input_label = self.mnist_train_labels_curr[self.mnist_train_index_curr:self.mnist_train_index_curr+self.batch_size*num_step,:].reshape((num_step,self.batch_size,1))
    chosen_ind = random.randint(0,num_step-2)
    batch_input2 = batch_input[chosen_ind,:,:,:,:] # query
    batch_output = batch_input_label[chosen_ind+1,:,:]
    self.mnist_train_index_curr += self.batch_size*num_step
    return [batch_input, batch_input2, batch_output, chosen_ind]
  
  def process(self,saver, sess, global_t, summary_writer, summary_op, loss_input):
    self.global_t = global_t
    while (self.global_t<=self.max_global_time_step):
      cur_lstm_states = self.network.get_initial_state()
      num_steps = LOCAL_T_MAX
      [batch_input, batch_input2, batch_output, chosen_ind] = self.generate_input_output(num_steps)

      batch_input = batch_input.reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,1))
      batch_input2 = batch_input2.reshape((self.batch_size,INPUT_SIZE,INPUT_SIZE,1))
      batch_output = batch_output.reshape(self.batch_size)

      if (self.global_t-self.prev_global_t_vis>=SAVE_VIS_INTERVAL):
        self.prev_global_t_vis = self.global_t         
        self.vis_save(sess, self.global_t)

      if (self.global_t-self.prev_global_t_model>=SAVE_MODEL_INTERVAL):
        self.prev_global_t_model = self.global_t         
        saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = self.global_t)

      cur_learning_rate = self._anneal_learning_rate(self.global_t)
      if ((self.global_t-self.prev_global_t_log) >= LOG_INTERVAL):
        self.prev_global_t_log = self.global_t
        list_to_run = [self.network.loss]
        list_keys = self.network.lstm_state_ph_list + [self.network.gt, self.network.input, self.network.input2, self.network.step_size]
        list_vals = cur_lstm_states + [batch_output, batch_input, batch_input2, [num_steps]]
        list_output = sess.run( list_to_run,
                feed_dict = dict(zip(list_keys, list_vals)) )
        loss_output_total = list_output[0]
       
        print("Step {}: loss = {}".format(self.global_t,loss_output_total))

        self._record_params(sess, summary_writer, summary_op, loss_input, loss_output_total, self.global_t)

      list_to_run = [self.apply_gradients]
      list_keys = self.network.lstm_state_ph_list + [self.network.gt, self.network.input, self.network.input2, self.network.step_size, self.learning_rate_input]
      list_vals = cur_lstm_states + [batch_output, batch_input, batch_input2, [num_steps], cur_learning_rate]
      list_output = sess.run( list_to_run,
                feed_dict = dict(zip(list_keys, list_vals)) )
       
      self.global_t += num_steps

    return self.global_t
    
