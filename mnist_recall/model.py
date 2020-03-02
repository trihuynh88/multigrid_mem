# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from constants import BATCH_SIZE
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
      self.gt = tf.placeholder("int32",[None])
      self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.gt, logits = self.output)) 

  def get_vars(self):
    raise NotImplementedError()

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

class MNISTRecallModel(Model):
  def __init__(self,
               thread_index,
               device="/cpu:0" ):
    Model.__init__(self, thread_index, device)

    input_size = INPUT_SIZE
    flattened_len = 3*3*10

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      variables_before = tf.trainable_variables()
      self.W_fc_last, self.b_fc_last = self._fc_variable([flattened_len, 10])
      self.input = tf.placeholder("float", [None, input_size, input_size, 1])
      self.input2 = tf.placeholder("float", [None, input_size, input_size, 1])   # query
      self.step_size = tf.placeholder(tf.float32, [1])
      self.lstm_state_list = []
      self.lstm_state_ph_list = []
      self.grid_dim_list = []
      input3 = tf.image.resize_images(self.input, (3,3), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input6 = tf.image.resize_images(self.input, (6,6), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input12 = tf.image.resize_images(self.input, (12,12), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input3scales = [input3, input6, input12]
      output_dims, self.output_grids1, lstm_states, lstm_states_ph = self.build_next_layer_arbit(input3scales, 0, 2, 0, 2, 4, 1)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids2, lstm_states, lstm_states_ph = self.build_next_layer_arbit(self.output_grids1, 0, 2, 0, 2, 4, 2)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids3, lstm_states, lstm_states_ph = self.build_next_layer_arbit(self.output_grids2, 0, 2, 0, 2, 8, 3)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      self.output_grids3[-1] = self.create_residual_conn_extra(self.output_grids1[-1], self.output_grids3[-1], 4, 8)
      self.output_grids3[-2] = self.create_residual_conn_extra(self.output_grids1[-2], self.output_grids3[-2], 4, 8)
      self.output_grids3[-3] = self.create_residual_conn_extra(self.output_grids1[-3], self.output_grids3[-3], 4, 8)

      output_dims, self.output_grids4, lstm_states, lstm_states_ph = self.build_next_layer_arbit(self.output_grids3, 0, 2, 0, 2, 8, 4)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      output_dims, self.output_grids5, lstm_states, lstm_states_ph = self.build_next_layer_arbit(self.output_grids4, 0, 2, 0, 2, 16, 5)
      self.lstm_state_list = self.lstm_state_list + lstm_states
      self.lstm_state_ph_list = self.lstm_state_ph_list + lstm_states_ph
      self.grid_dim_list = self.grid_dim_list + output_dims

      self.output_grids5[-1] = self.create_residual_conn_extra(self.output_grids3[-1], self.output_grids5[-1], 8, 16)
      self.output_grids5[-2] = self.create_residual_conn_extra(self.output_grids3[-2], self.output_grids5[-2], 8, 16)
      self.output_grids5[-3] = self.create_residual_conn_extra(self.output_grids3[-3], self.output_grids5[-3], 8, 16)

      input3_2 = tf.image.resize_images(self.input2, (3,3), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input6_2 = tf.image.resize_images(self.input2, (6,6), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input12_2 = tf.image.resize_images(self.input2, (12,12), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      input3scales_2 = [input3_2, input6_2, input12_2]

      output_dims, output_grids1_2 = self.build_next_layer_convo_arbit(input3scales_2, 0, 2, 0, 2, 1, 4, 1) 
      for scale_ind in range(3):
        output_grids1_2[scale_ind] = tf.concat([output_grids1_2[scale_ind],self.output_grids1[scale_ind][-BATCH_SIZE:,:,:,:]],-1)

      output_dims, output_grids2_2 = self.build_next_layer_convo_arbit(output_grids1_2, 0,2,0,2,8,4, 2)
      for scale_ind in range(3):
        output_grids2_2[scale_ind] = tf.concat([output_grids2_2[scale_ind],self.output_grids2[scale_ind][-BATCH_SIZE:,:,:,:]],-1)

      output_dims, output_grids3_2 = self.build_next_layer_convo_arbit(output_grids2_2, 0,2,0,2,8,8, 3)
      output_grids3_2[-1] = self.create_residual_conn(output_grids1_2[-1], output_grids3_2[-1])
      output_grids3_2[-2] = self.create_residual_conn(output_grids1_2[-2], output_grids3_2[-2])
      output_grids3_2[-3] = self.create_residual_conn(output_grids1_2[-3], output_grids3_2[-3])

      for scale_ind in range(3):
        output_grids3_2[scale_ind] = tf.concat([output_grids3_2[scale_ind],self.output_grids3[scale_ind][-BATCH_SIZE:,:,:,:]],-1)

      output_dims, output_grids4_2 = self.build_next_layer_convo_arbit(output_grids3_2, 0,2,0,2,16,8, 4)
      for scale_ind in range(3):
        output_grids4_2[scale_ind] = tf.concat([output_grids4_2[scale_ind],self.output_grids4[scale_ind][-BATCH_SIZE:,:,:,:]],-1)

      output_dims, output_grids5_2 = self.build_next_layer_convo_arbit(output_grids4_2, 0,2,0,2,16,16, 5)
      output_grids5_2[-1] = self.create_residual_conn(output_grids3_2[-1], output_grids5_2[-1])
      output_grids5_2[-2] = self.create_residual_conn(output_grids3_2[-2], output_grids5_2[-2])
      output_grids5_2[-3] = self.create_residual_conn(output_grids3_2[-3], output_grids5_2[-3])

      for scale_ind in range(3):
        output_grids5_2[scale_ind] = tf.concat([output_grids5_2[scale_ind],self.output_grids5[scale_ind][-BATCH_SIZE:,:,:,:]],-1)

      output_dims, output_grids10_2 = self.build_next_layer_convo(output_grids5_2, 3, 32, 16, 10)

      output_dims, output_grids11_2 = self.build_next_layer_convo(output_grids10_2, 3, 16, 16, 11)
      output_dims, output_grids12_2 = self.build_next_layer_convo(output_grids11_2, 2, 16, 16, 12)

      output_dims, output_grids13_2 = self.build_next_layer_convo(output_grids12_2, 2, 16, 16, 13)

      output_grids13_2[0] = self.create_residual_conn(output_grids11_2[0], output_grids13_2[0])
      output_grids13_2[1] = self.create_residual_conn(output_grids11_2[1], output_grids13_2[1])

      output_dims, output_grids14_2 = self.build_next_layer_convo(output_grids13_2, 1, 16, 10, 14)

      output_grids14_2_flat = tf.reshape(output_grids14_2[0], [-1, flattened_len])
      output_grids14_2_last = (tf.matmul(output_grids14_2_flat, self.W_fc_last) + self.b_fc_last)
      self.output = tf.reshape(output_grids14_2_last,[-1,10])
      self.prediction = tf.nn.softmax(self.output)

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

      scope_name = "conv_" + str(lay_ind) + "_" + str(i)
      output_grid = tf.contrib.layers.batch_norm(output_grid,
                                          center=True, scale=True,
                                          scope=scope_name)

      output_grids.append(output_grid)
    return output_dims, output_grids

  def build_next_layer_arbit(self, prev_grids, prev_start_level, prev_end_level, cur_start_level, cur_end_level, output_feature_chan, lay_ind):
    num_input_grids = len(prev_grids)
    output_grids = []
    lstm_states = []
    lstm_states_ph = []
    output_dims = []

    for i in range(cur_start_level,cur_end_level+1):
      output_dim = (3*(2**i),3*(2**i),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      iprev = i-prev_start_level
      if (iprev-1)>=0 and (iprev-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[iprev-1].shape[1], prev_grids[iprev-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[iprev-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
      if iprev>=0 and iprev<num_input_grids:
        if concat_grid == None:
          concat_grid = prev_grids[iprev]
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[iprev]], 3)
      if iprev+1>=0 and iprev+1<num_input_grids:
        prev_down = tf.layers.max_pooling2d(prev_grids[iprev+1],[2,2],2)
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
        lstm_outputs = tf.contrib.layers.batch_norm(lstm_outputs,
                                          center=True, scale=True,
                                          scope=scope_name)

      output_grids.append(lstm_outputs)
      lstm_states.append(lstm_state)
      lstm_states_ph.append(initial_lstm_state)
    return output_dims, output_grids, lstm_states, lstm_states_ph

  def build_next_layer_convo_arbit(self, prev_grids, prev_start_level, prev_end_level, cur_start_level, cur_end_level, input_feature_chan, output_feature_chan, lay_ind):
    num_input_grids = len(prev_grids)
    output_grids = []
    output_dims = []

    for i in range(cur_start_level, cur_end_level+1):
      sum_input_feature_chan = 0
      output_dim = (3*(2**i),3*(2**i),output_feature_chan)
      output_dims.append(output_dim)
      concat_grid = None
      iprev = i-prev_start_level
      if (iprev-1)>=0 and (iprev-1)<num_input_grids:
        prev_spatial_dim = (prev_grids[iprev-1].shape[1], prev_grids[iprev-1].shape[2])
        next_spatial_dim = (prev_spatial_dim[0]*2, prev_spatial_dim[1]*2)
        prev_up = tf.image.resize_images(prev_grids[iprev-1], next_spatial_dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_grid = prev_up
        sum_input_feature_chan = input_feature_chan
      if iprev>=0 and iprev<num_input_grids:
        if concat_grid == None:   
          concat_grid = prev_grids[iprev]
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_grids[iprev]], 3)
          sum_input_feature_chan += input_feature_chan
      if iprev+1>=0 and iprev+1<num_input_grids:   
        prev_down = tf.layers.max_pooling2d(prev_grids[iprev+1],[2,2],2)
        if concat_grid == None:
          concat_grid = prev_down
          sum_input_feature_chan = input_feature_chan
        else:
          concat_grid = tf.concat([concat_grid,prev_down],3)
          sum_input_feature_chan += input_feature_chan
      W_conv1_m , b_conv1_m = self._conv_variable([3, 3, sum_input_feature_chan, output_feature_chan])
      output_grid = tf.nn.relu(self._conv2d(concat_grid,  W_conv1_m, 1) + b_conv1_m)

      scope_name = "conv_" + str(lay_ind) + "_" + str(i)
      output_grid = tf.contrib.layers.batch_norm(output_grid,
                                          center=True, scale=True,
                                          scope=scope_name)


      output_grids.append(output_grid)
    return output_dims, output_grids

  def get_initial_state(self):
    lstm_state_out = []
    for i in range(len(self.lstm_state_list)):
      cur_dim = self.grid_dim_list[i]
      lstm_state_out.append(tf.contrib.rnn.LSTMStateTuple(np.zeros([BATCH_SIZE, cur_dim[0], cur_dim[1], cur_dim[2]]),
                                                        np.zeros([BATCH_SIZE, cur_dim[0], cur_dim[1], cur_dim[2]])))

    return lstm_state_out

  def get_vars(self):
    return tf.trainable_variables()

