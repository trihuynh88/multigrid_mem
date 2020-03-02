# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import time
import sys

from model import LocalizationModel
from map_env import MapEnv 

from constants import LOCAL_T_MAX
from constants import GRAD_NORM_CLIP
from constants import BATCH_SIZE
from constants import LOG_FILE
from constants import MAP_SIZE
from constants import SAVE_INTERVAL
from constants import SAVE_VIS_INTERVAL
from constants import CHECKPOINT_DIR
from constants import END_DECAY_TIME
from constants import END_ALPHA
from constants import INPUT_SIZE
from constants import GENERIC_MAP_SIZE
from constants import LOG_INTERVAL

import pdb

import sys
import cv2
import os
import copy
from random import shuffle

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
    self.game_state = MapEnv(self.batch_size,GENERIC_MAP_SIZE)
    self.game_state.reset_agent_pos()
    self.game_step = 0

    self.initial_learning_rate = initial_learning_rate

    self.batch_ind,self.row_ind,self.col_ind = np.indices((self.batch_size,INPUT_SIZE,INPUT_SIZE))
    self.network = network

  
  def vis_convert_to_image(self,matrix, env_state):
    im = np.zeros((matrix.shape[0],matrix.shape[1],3))
    #background cells in the combined visualization
    background_color = (0.5,0.5,0.5)
    indices = np.where(matrix == -1)
    im[indices[0],indices[1],:] = background_color
    #unknown cells
    unknown_color = (0,0,0)
    indices = np.where(matrix == env_state.unknown_mark)
    im[indices[0],indices[1],:] = unknown_color
    #Wall cells
    wall_color = (0,0,1)
    indices = np.where(matrix == env_state.unavail_mark)
    im[indices[0],indices[1],:] = wall_color
    #Free cells
    free_color = (1,1,1)
    indices = np.where(matrix == env_state.free_mark)
    im[indices[0],indices[1],:] = free_color
    #Visited cells
    visited_color = (0.5,0.5,1)
    indices = np.where(matrix == env_state.visited_mark)
    im[indices[0],indices[1],:] = visited_color
    #Agent cell
    agent_color = (1,0,0)
    indices = np.where(matrix == env_state.agent_mark)
    im[indices[0],indices[1],:] = agent_color  
    return im

  def vis_save(self,sess,timestep):
    vis_dir = CHECKPOINT_DIR+("/%d"%timestep)
    if os.path.exists(vis_dir):
      return
    os.mkdir(vis_dir)
    MAP_SIZE_MEM = 48
    episode_end = False
    cur_lstm_states = self.network.get_initial_state()
    step_ind = -1
    env_state = MapEnv(self.batch_size,MAP_SIZE*2)
    env_state.reset_agent_pos()
    while not episode_end:
      batch_map_input = []
      batch_map_input2 = []
      batch_loc_output = []
      step_ind += 1
      map_input = self.convert_maps(env_state.get_input_ego(),env_state.generic_pos) 
      loc = np.copy(env_state.agent_pos)
      loc_generic = np.copy(env_state.generic_pos)
      map_input_noloc = self.remove_loc(map_input)
      loc_output =  self.get_loc_output(map_input, env_state)
 
      full_maps = np.copy(env_state.maps)
      full_maps_generic = np.copy(env_state.generic_maps)
      episode_end = env_state.move_spiral()
      num_steps = 1
      batch_map_input.append(map_input)
      batch_map_input2.append(map_input_noloc)
      batch_loc_output.append(loc_output)
   
      batch_map_input = np.asarray(batch_map_input).reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,4))
      batch_map_input2 = np.asarray(batch_map_input2).reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,4))
      batch_loc_output = np.asarray(batch_loc_output).reshape((num_steps*self.batch_size,MAP_SIZE_MEM*MAP_SIZE_MEM))

      list_to_run = [self.network.locpred] + self.network.lstm_state_list + self.network.output_grids7
      list_keys = self.network.lstm_state_ph_list + [self.network.ma, self.network.ma2, self.network.step_size]
      list_vals = cur_lstm_states + [batch_map_input, batch_map_input2, [num_steps]]
      list_output = sess.run( list_to_run,
                              feed_dict = dict(zip(list_keys, list_vals))        
                    )
      pred_maps = list_output[0][:batch_map_input.shape[0]]
      cur_lstm_states = list_output[1:1+len(self.network.lstm_state_list)]
      learned_maps = list_output[1+len(self.network.lstm_state_list):]
      gt_data_list = batch_loc_output.reshape((self.batch_size,MAP_SIZE_MEM,MAP_SIZE_MEM))
      locpred_gt = loc
      locpred_gt_generic = loc_generic
      pred_maps = pred_maps.reshape((self.batch_size,MAP_SIZE_MEM,MAP_SIZE_MEM))

      num_vis_env = 2
      for ind_vis_env in range(num_vis_env):
        pred_map = pred_maps[ind_vis_env].reshape((MAP_SIZE_MEM,MAP_SIZE_MEM))
        pred_map_argmax = (pred_map>=0.5)
        gt_data = gt_data_list[ind_vis_env].reshape((MAP_SIZE_MEM,MAP_SIZE_MEM))
        gt_data_argmax = gt_data         
        full_map_data = full_maps[ind_vis_env]
        full_map_data_generic = full_maps_generic[ind_vis_env]
        cur_agent_pos = locpred_gt[ind_vis_env]
        cur_agent_pos_generic = locpred_gt_generic[ind_vis_env]
        full_map_data[cur_agent_pos[0],cur_agent_pos[1]] = env_state.agent_mark
        full_map_data_generic[cur_agent_pos_generic[0],cur_agent_pos_generic[1]] = env_state.agent_mark
        map_input = batch_map_input[ind_vis_env,:,:,:].reshape((INPUT_SIZE,INPUT_SIZE,4))
        map_input_argmax = map_input[:,:,:2].argmax(2).reshape((INPUT_SIZE, INPUT_SIZE))

 
        step_string = ("%02d_%04d"%(ind_vis_env, step_ind))
        step_path = vis_dir + "/" + step_string + ".jpg"
  
        vis_width = MAP_SIZE+2*MAP_SIZE+5+20+3*MAP_SIZE_MEM+INPUT_SIZE
        vis_height = MAP_SIZE*2 + 11 + MAP_SIZE_MEM*2+3 
        final_image = np.ones((vis_height,vis_width))*(-1)
  
        final_image[:MAP_SIZE,:MAP_SIZE] = full_map_data
        final_image[:MAP_SIZE*2,MAP_SIZE+5:MAP_SIZE+5+MAP_SIZE*2] = full_map_data_generic
        final_image[MAP_SIZE-INPUT_SIZE/2:MAP_SIZE+INPUT_SIZE/2+1,MAP_SIZE+5+MAP_SIZE*2+5:MAP_SIZE+5+MAP_SIZE*2+5+INPUT_SIZE] = map_input_argmax
        
  
        scale_factor = 10
        map_img_scaled = np.kron(final_image, np.ones((scale_factor,scale_factor)))
  
        map_img_vis = (self.vis_convert_to_image(map_img_scaled, env_state)*255).astype(np.uint8)

        lstm_image = np.ones((MAP_SIZE_MEM+3+MAP_SIZE_MEM,vis_width))*(-1)
        lstm_state = cur_lstm_states[0]
        lstm_state_vis = []
        cur_lstm_states_vis = cur_lstm_states[-5:]
        for list_lstm_ind in range(len(cur_lstm_states_vis)):
          cur_lstm = []
          for lstm_ind in range(2):
            cur_mean = np.expand_dims(np.mean(cur_lstm_states_vis[list_lstm_ind][lstm_ind],axis=-1),axis=-1)
            cur_var = np.expand_dims(np.var(cur_lstm_states_vis[list_lstm_ind][lstm_ind],axis=-1),axis=-1)
            cur_lstm.append(np.concatenate((cur_mean,cur_var),axis=-1))
          lstm_state_vis.append(cur_lstm)

        start_lstm_pos = 0
        for list_lstm_ind in range(len(cur_lstm_states_vis)):
          cur_lstm_size = 3*(2**list_lstm_ind)
          for lstm_ind in range(2):
            for lstm_chan in range(2):
              lstm_image[lstm_ind*(cur_lstm_size+3):lstm_ind*(cur_lstm_size+3)+cur_lstm_size,start_lstm_pos+lstm_chan*(cur_lstm_size+5):start_lstm_pos+lstm_chan*(cur_lstm_size+5)+cur_lstm_size] = lstm_state_vis[list_lstm_ind][lstm_ind][ind_vis_env,:,:,lstm_chan]
          start_lstm_pos += (cur_lstm_size*2+10)


        lstm_image = (lstm_image+1)/2.0
        lstm_img_scaled = np.kron(lstm_image, np.ones((scale_factor,scale_factor)))
        lstm_img_scaled = (lstm_img_scaled*255).astype(np.uint8)
        for chan_ind in range(3):
          map_img_vis[-lstm_img_scaled.shape[0]:,:,chan_ind] = lstm_img_scaled
        
        pred_map_scaled = np.kron(pred_map*1.0/np.amax(pred_map), np.ones((scale_factor,scale_factor)))
        pred_map_scaled = (pred_map_scaled*255).astype(np.uint8)
        for chan_ind in range(3):
          map_img_vis[:MAP_SIZE_MEM*10,(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5)*10:(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5+MAP_SIZE_MEM)*10,chan_ind] = pred_map_scaled

        pred_argmax_scaled = np.kron(pred_map_argmax.astype(np.int32), np.ones((scale_factor,scale_factor)))
        pred_argmax_scaled = (pred_argmax_scaled*255).astype(np.uint8)
        for chan_ind in range(3):
          map_img_vis[:MAP_SIZE_MEM*10,(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5+MAP_SIZE_MEM+5)*10:(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5+MAP_SIZE_MEM+5+MAP_SIZE_MEM)*10,chan_ind] = pred_argmax_scaled

        gt_argmax_scaled = np.kron(gt_data_argmax, np.ones((scale_factor,scale_factor)))
        gt_argmax_scaled = (gt_argmax_scaled*255).astype(np.uint8)
        for chan_ind in range(3):
          map_img_vis[:MAP_SIZE_MEM*10,(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5+MAP_SIZE_MEM+5+MAP_SIZE_MEM+5)*10:(MAP_SIZE+5+INPUT_SIZE+5+MAP_SIZE*2+5+MAP_SIZE_MEM+5+MAP_SIZE_MEM+5+MAP_SIZE_MEM)*10,chan_ind] = gt_argmax_scaled

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.6
        fontColor              = (255,255,255)
        lineType               = 2
 
        saved_image = cv2.cvtColor(map_img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(step_path,saved_image)

  def _anneal_learning_rate(self, global_time_step):
    if global_time_step<END_DECAY_TIME:
      learning_rate = self.initial_learning_rate+(END_ALPHA-self.initial_learning_rate) * global_time_step / END_DECAY_TIME
    else:
      learning_rate = END_ALPHA
    return learning_rate


  def _record_params(self, sess, summary_writer, summary_op, loss_input, loss, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      loss_input: np.asscalar(loss),
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def convert_maps(self, maps_input, loc):
    maps_output = np.zeros((maps_input.shape[0],maps_input.shape[1],maps_input.shape[2],4))
    maps_output[self.batch_ind,self.row_ind,self.col_ind,maps_input[self.batch_ind,self.row_ind,self.col_ind]] = 1 
    for i in range(maps_input.shape[0]):
      maps_output[i,:,:,2] = loc[i,0]
      maps_output[i,:,:,3] = loc[i,1]
    return maps_output  
 
  def remove_loc(self, maps_input):
    maps_output = np.copy(maps_input)
    maps_output[:,:,:,2] = 0
    maps_output[:,:,:,3] = 0
    return maps_output
 
  def get_loc_output(self, pattern, env_state):
    output = np.zeros((self.batch_size,48*48))
    pattern = pattern[:,:,:,:2].argmax(-1)
    for i in range(self.batch_size):
      cur_pattern = pattern[i]
      output[i] = np.copy(env_state.pattern_loc[i][env_state.to_tuple(cur_pattern)]).reshape(48*48)
    return output
        
   
  def process(self,saver, sess, global_t, summary_writer, summary_op, loss_input):
    self.global_t = global_t
    # variable controling output logging
    self.prev_global_t = global_t
    self.prev_global_t_log = global_t
    self.prev_global_t_vis = global_t

    self.start_time = time.time() 
    stop = False
    while not stop: 
      episode_end = False
      cur_lstm_states = self.network.get_initial_state()
      self.game_state.reset_agent_pos()
      list_map_input_all = []
      while not episode_end: 
        if (self.global_t>self.max_global_time_step):
          stop = True
          break
        num_steps = 0
        batch_map_input = []
        batch_loc_output = []
        batch_map_input2 = []
        batch_loc_output2 = []
        while (not episode_end) and (num_steps<LOCAL_T_MAX):
          map_input = self.convert_maps(self.game_state.get_input_ego(), self.game_state.generic_pos) 
          episode_end = self.game_state.move_spiral()
          num_steps += 1
          batch_map_input.append(map_input)
          list_map_input_all.append(self.remove_loc(map_input))
          chosen_ind = random.randint(0,len(list_map_input_all)-1) 
          batch_map_input2.append(list_map_input_all[chosen_ind])
          batch_loc_output2.append(self.get_loc_output(list_map_input_all[chosen_ind], self.game_state))         
   
        batch_map_input = np.asarray(batch_map_input).reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,4))
        batch_map_input2 = np.asarray(batch_map_input2).reshape((num_steps*self.batch_size,INPUT_SIZE,INPUT_SIZE,4))
        batch_loc_output2 = np.asarray(batch_loc_output2).reshape((num_steps*self.batch_size,48*48))
        if (self.global_t-self.prev_global_t>=SAVE_INTERVAL):
          self.prev_global_t = self.global_t
          saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = self.global_t)

        if (self.global_t-self.prev_global_t_vis>=SAVE_VIS_INTERVAL):
          self.prev_global_t_vis = self.global_t
          self.vis_save(sess,self.global_t)

        cur_learning_rate = self._anneal_learning_rate(self.global_t)
        if ((self.global_t-self.prev_global_t_log) >= LOG_INTERVAL):
          self.prev_global_t_log = self.global_t
          list_to_run = [self.network.loss]
          list_keys = self.network.lstm_state_ph_list + [self.network.locpred_gt, self.network.ma, self.network.ma2, self.network.step_size]
          list_vals = cur_lstm_states + [batch_loc_output2, batch_map_input, batch_map_input2, [num_steps]]
          list_output = sess.run( list_to_run,
                  feed_dict = dict(zip(list_keys, list_vals)) 
               )
          loss_output_total = list_output[0]
                   
          print("Step {}: loss = {}, lr = {}".format(self.global_t,loss_output_total, cur_learning_rate))

          self._record_params(sess, summary_writer, summary_op, loss_input, loss_output_total, self.global_t)

        list_to_run = [self.apply_gradients] + self.network.lstm_state_list
        list_keys = self.network.lstm_state_ph_list + [self.network.locpred_gt, self.network.ma, self.network.ma2, self.network.step_size, self.learning_rate_input]
        list_vals = cur_lstm_states + [batch_loc_output2, batch_map_input, batch_map_input2, [num_steps], cur_learning_rate]
        list_output = sess.run( list_to_run,
                                feed_dict = dict(zip(list_keys, list_vals)) 
                      )
        cur_lstm_states = list_output[1:]
         
        self.global_t += num_steps

    return self.global_t
    
