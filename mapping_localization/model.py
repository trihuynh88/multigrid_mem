# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from constants import BATCH_SIZE
from constants import MAP_SIZE
from constants import INPUT_SIZE

import pdb

class Model(object):
  def __init__(self,
               thread_index,
               device="/cpu:0"):
    self._thread_index = thread_index
    self._device = device    

  def prepare_loss(self):
    with tf.device(self._device):

      self.locpred_gt = tf.placeholder("float",[None,48*48])

      self.locpred_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.locpred_gt, logits = self.multigrid_output2)) 

      self.loss = self.locpred_loss 

  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")


class LocalizationModel(Model):
  def __init__(self,
               thread_index,
               device="/cpu:0" ):
    Model.__init__(self, thread_index, device)

    map_size = INPUT_SIZE
    flattened_len = 3*3*16

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      variables_before = tf.trainable_variables()
      self.ma = tf.placeholder("float", [None, map_size, map_size, 4])
      self.ma2 = tf.placeholder("float", [None, map_size, map_size, 4])
      self.step_size = tf.placeholder(tf.float32, [1])
      self.lstm_state_list = []
      self.lstm_state_ph_list = []
      self.grid_dim_list = []
      output_dims, self.output_grids1, lstm_states, lstm_states_ph = self.build_next_layer([self.ma], 2, 4, 1)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids2, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids1, 2, 4, 2)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids3, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids2, 3, 8, 3)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims


      self.output_grids3[0] = self.create_residual_conn_extra(self.output_grids1[0], self.output_grids3[0], 4, 8)
      self.output_grids3[1] = self.create_residual_conn_extra(self.output_grids1[1], self.output_grids3[1], 4, 8)

      output_dims, self.output_grids4, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids3, 3, 8, 4)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids5, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids4, 4, 16, 5)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      self.output_grids5[0] = self.create_residual_conn_extra(self.output_grids3[0], self.output_grids5[0], 8, 16)
      self.output_grids5[1] = self.create_residual_conn_extra(self.output_grids3[1], self.output_grids5[1], 8, 16)
      self.output_grids5[2] = self.create_residual_conn_extra(self.output_grids3[2], self.output_grids5[2], 8, 16)

      output_dims, self.output_grids6, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids5, 4, 16, 6)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims


      output_dims, self.output_grids7, lstm_states, lstm_states_ph = self.build_next_layer(self.output_grids6, 5, 16, 7)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      self.output_grids7[0] = self.create_residual_conn(self.output_grids5[0], self.output_grids7[0])
      self.output_grids7[1] = self.create_residual_conn(self.output_grids5[1], self.output_grids7[1])
      self.output_grids7[2] = self.create_residual_conn(self.output_grids5[2], self.output_grids7[2])
      self.output_grids7[3] = self.create_residual_conn(self.output_grids5[3], self.output_grids7[3])

      output_dims, output_grids1_2 = self.build_next_layer_convo([self.ma2], 2, 4, 4, 1)
      for scale_ind in range(2):
        output_grids1_2[scale_ind] = tf.concat([output_grids1_2[scale_ind],self.output_grids1[scale_ind]],-1)

      output_dims, output_grids2_2 = self.build_next_layer_convo(output_grids1_2, 2, 8, 4, 2)
      for scale_ind in range(2):
        output_grids2_2[scale_ind] = tf.concat([output_grids2_2[scale_ind],self.output_grids2[scale_ind]],-1)

      output_dims, output_grids3_2 = self.build_next_layer_convo(output_grids2_2, 3, 8, 8, 3)
      output_grids3_2[0] = self.create_residual_conn(output_grids1_2[0], output_grids3_2[0])
      output_grids3_2[1] = self.create_residual_conn(output_grids1_2[1], output_grids3_2[1])
      for scale_ind in range(3):
        output_grids3_2[scale_ind] = tf.concat([output_grids3_2[scale_ind],self.output_grids3[scale_ind]],-1)

      output_dims, output_grids4_2 = self.build_next_layer_convo(output_grids3_2, 3, 16, 8, 4)
      for scale_ind in range(3):
        output_grids4_2[scale_ind] = tf.concat([output_grids4_2[scale_ind],self.output_grids4[scale_ind]],-1)

      output_dims, output_grids5_2 = self.build_next_layer_convo(output_grids4_2, 4, 16, 16, 5)
      output_grids5_2[0] = self.create_residual_conn(output_grids3_2[0], output_grids5_2[0])
      output_grids5_2[1] = self.create_residual_conn(output_grids3_2[1], output_grids5_2[1])
      output_grids5_2[2] = self.create_residual_conn(output_grids3_2[2], output_grids5_2[2])
      for scale_ind in range(4):
        output_grids5_2[scale_ind] = tf.concat([output_grids5_2[scale_ind],self.output_grids5[scale_ind]],-1)

      output_dims, output_grids6_2 = self.build_next_layer_convo(output_grids5_2, 4, 32, 16, 6)
      for scale_ind in range(4):
        output_grids6_2[scale_ind] = tf.concat([output_grids6_2[scale_ind],self.output_grids6[scale_ind]],-1)

      output_dims, output_grids7_2 = self.build_next_layer_convo(output_grids6_2, 5, 32, 16, 7)
      for scale_ind in range(5):
        output_grids7_2[scale_ind] = tf.concat([output_grids7_2[scale_ind],self.output_grids7[scale_ind]],-1)


      output_dims, output_grids8_2 = self.build_next_layer_convo_reverse(output_grids7_2, 4, 32, 16,4, 8)
      output_dims, output_grids9_2 = self.build_next_layer_convo_reverse(output_grids8_2, 4, 16, 16,4, 9)
      output_dims, output_grids10_2 = self.build_next_layer_convo_reverse(output_grids9_2, 3, 16, 16,4, 10)

      output_dims, output_grids11_2 = self.build_next_layer_convo_reverse(output_grids10_2, 3, 16, 16,4, 11)
      output_grids11_2[-1] = self.create_residual_conn(output_grids9_2[-1], output_grids11_2[-1])
      output_grids11_2[-2] = self.create_residual_conn(output_grids9_2[-2], output_grids11_2[-2])
      output_grids11_2[-3] = self.create_residual_conn(output_grids9_2[-3], output_grids11_2[-3])


      output_dims, output_grids12_2 = self.build_next_layer_convo_reverse(output_grids11_2, 2, 16, 16, 4,12)

      output_dims, output_grids13_2 = self.build_next_layer_convo_reverse(output_grids12_2, 2, 16, 16,4, 13)

      output_grids13_2[-1] = self.create_residual_conn(output_grids11_2[-1], output_grids13_2[-1])
      output_grids13_2[-2] = self.create_residual_conn(output_grids11_2[-2], output_grids13_2[-2])

      output_dims, output_grids14_2 = self.build_next_layer_convo_reverse_without_activation(output_grids13_2, 1, 16, 1,4, 14)

      self.multigrid_output2 = tf.reshape(output_grids14_2[0],[-1, 48*48])
      self.locpred = tf.sigmoid(self.multigrid_output2)

      scope.reuse_variables()
      self.variables = sorted(list(set(tf.trainable_variables()) - set(variables_before)), key=lambda x:x.name)

  def create_residual_conn(self,first_elem,second_elem):
    return first_elem + second_elem
  def create_residual_conn_extra(self,first_elem,second_elem,first_dim,second_dim):
    first_elem_padded = tf.pad(first_elem, tf.constant([[0,0],[0,0],[0,0],[0,second_dim-first_dim]]), "CONSTANT")
    return first_elem_padded + second_elem

  def build_next_layer_convo(self, prev_grids, num_output_grids, input_feature_chan, output_feature_chan, lay_ind):
    num_input_grids = len(prev_grids)
    output_grids = []
    output_dims = []

    for i in range(num_output_grids):
      sum_input_feature_chan = 0
      output_dim = (3*(2**i),3*(2**i),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      if (i-1)>=0 and (i-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[i-1].shape[1], prev_grids[i-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[i-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
        sum_input_feature_chan = input_feature_chan
      if i>=0 and i<num_input_grids:
        if concat_grid == None:
          concat_grid = prev_grids[i]
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[i]], 3)
          sum_input_feature_chan += input_feature_chan
      if i+1>=0 and i+1<num_input_grids:
        prev_down = tf.layers.max_pooling2d(prev_grids[i+1],[2,2],2)
        if concat_grid == None:
          concat_grid = prev_down
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_down],3)
          sum_input_feature_chan += input_feature_chan
      W_conv1_m , b_conv1_m = self._conv_variable([3, 3, sum_input_feature_chan, output_feature_chan])
      output_grid = tf.nn.relu(self._conv2d(concat_grid,  W_conv1_m, 1) + b_conv1_m)
      output_grids.append(output_grid)
    return output_dims, output_grids

  def build_next_layer_convo_reverse(self, prev_grids, num_output_grids, input_feature_chan, output_feature_chan, largest_scale, lay_ind):
    num_input_grids = len(prev_grids)
    output_grids = []
    output_dims = []

    for i in range(num_output_grids):
      j = (num_input_grids - num_output_grids) + i
      sum_input_feature_chan = 0
      scale = largest_scale - num_output_grids + 1 + i
      output_dim = (3*(2**scale),3*(2**scale),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      if (j-1)>=0 and (j-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[j-1].shape[1], prev_grids[j-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[j-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
        sum_input_feature_chan = input_feature_chan
      if j>=0 and j<num_input_grids:
        if concat_grid == None:
          concat_grid = prev_grids[j]
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[j]], 3)
          sum_input_feature_chan += input_feature_chan
      if j+1>=0 and j+1<num_input_grids:
        prev_down = tf.layers.max_pooling2d(prev_grids[j+1],[2,2],2)
        if concat_grid == None:
          concat_grid = prev_down
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_down],3)
          sum_input_feature_chan += input_feature_chan
      W_conv1_m , b_conv1_m = self._conv_variable([3, 3, sum_input_feature_chan, output_feature_chan])
      output_grid = tf.nn.relu(self._conv2d(concat_grid,  W_conv1_m, 1) + b_conv1_m)
      output_grids.append(output_grid)
    return output_dims, output_grids

  def build_next_layer_convo_reverse_without_activation(self, prev_grids, num_output_grids, input_feature_chan, output_feature_chan, largest_scale, lay_ind):
    num_input_grids = len(prev_grids) 
    output_grids = []
    output_dims = []

    for i in range(num_output_grids):
      j = (num_input_grids - num_output_grids) + i
      sum_input_feature_chan = 0
      scale = largest_scale - num_output_grids + 1 + i
      output_dim = (3*(2**scale),3*(2**scale),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      if (j-1)>=0 and (j-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[j-1].shape[1], prev_grids[j-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[j-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
        sum_input_feature_chan = input_feature_chan
      if j>=0 and j<num_input_grids:
        if concat_grid == None:
          concat_grid = prev_grids[j]
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[j]], 3)
          sum_input_feature_chan += input_feature_chan
      if j+1>=0 and j+1<num_input_grids:
        prev_down = tf.layers.max_pooling2d(prev_grids[j+1],[2,2],2)
        if concat_grid == None:
          concat_grid = prev_down
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_down],3)
          sum_input_feature_chan += input_feature_chan
      W_conv1_m , b_conv1_m = self._conv_variable([3, 3, sum_input_feature_chan, output_feature_chan])
      output_grid = (self._conv2d(concat_grid,  W_conv1_m, 1) + b_conv1_m)
      output_grids.append(output_grid)
    return output_dims, output_grids

  def build_next_layer(self, prev_grids, num_output_grids, output_feature_chan, lay_ind):
    num_input_grids = len(prev_grids)
    output_grids = []
    lstm_states = [] 
    lstm_states_ph = []
    output_dims = []

    for i in range(num_output_grids):
      output_dim = (3*(2**i),3*(2**i),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      if (i-1)>=0 and (i-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[i-1].shape[1], prev_grids[i-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[i-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
      if i>=0 and i<num_input_grids:
        if concat_grid == None:
          concat_grid = prev_grids[i]
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[i]], 3)
      if i+1>=0 and i+1<num_input_grids:
        prev_down = tf.layers.max_pooling2d(prev_grids[i+1],[2,2],2)
        if concat_grid == None:
          concat_grid = prev_down
        else:
          concat_grid = tf.concat([concat_grid,prev_down],3)
      concat_grid_reshaped = tf.reshape(concat_grid, [-1,BATCH_SIZE,concat_grid.shape[1],concat_grid.shape[2],concat_grid.shape[3]])
      lstm = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[concat_grid.shape[1],concat_grid.shape[2],concat_grid.shape[3]], kernel_shape=[3,3], output_channels=output_feature_chan)
      initial_lstm_state0 = tf.placeholder(tf.float32, [BATCH_SIZE, concat_grid.shape[1], concat_grid.shape[2], output_feature_chan])
      initial_lstm_state1 = tf.placeholder(tf.float32, [BATCH_SIZE, concat_grid.shape[1], concat_grid.shape[2], output_feature_chan])
      initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(initial_lstm_state0,
                                                              initial_lstm_state1)

      scope_name = "lstm_" + str(lay_ind) + "_" + str(i)
      with tf.variable_scope(scope_name) as scope:
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm,
                                                        concat_grid_reshaped,
                                                        initial_state = initial_lstm_state,
                                                        sequence_length = tf.fill([BATCH_SIZE],self.step_size[0]),
                                                        time_major = True,
                                                        scope = scope)
      lstm_outputs = tf.reshape(lstm_outputs, [-1, concat_grid.shape[1], concat_grid.shape[2], output_feature_chan])
      output_grids.append(lstm_outputs)
      lstm_states.append(lstm_state)
      lstm_states_ph.append(initial_lstm_state)
    return output_dims, output_grids, lstm_states, lstm_states_ph




  def get_initial_state(self):
    lstm_state_out = []
    for i in range(len(self.lstm_state_list)):
      cur_dim = self.grid_dim_list[i]
      lstm_state_out.append(tf.contrib.rnn.LSTMStateTuple(np.zeros([BATCH_SIZE, cur_dim[0], cur_dim[1], cur_dim[2]]),
                                                        np.zeros([BATCH_SIZE, cur_dim[0], cur_dim[1], cur_dim[2]])))

    return lstm_state_out

  def get_vars(self):
    return tf.trainable_variables()

