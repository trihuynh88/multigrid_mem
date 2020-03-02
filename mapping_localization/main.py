# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from model import LocalizationModel
from training_thread import TrainingThread
from rmsprop_applier import RMSPropApplier

from constants import MAX_TIME_STEP
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import SAVE_INTERVAL
from constants import INITIAL_ALPHA
from constants import CHECKPOINT_DIR

import pdb
import sys



def count_number_trainable_params():
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params


device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = INITIAL_ALPHA

global_t = 0

stop_requested = False

network = LocalizationModel(-1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

training_thread = TrainingThread(network,initial_learning_rate,learning_rate_input,grad_applier,MAX_TIME_STEP,device)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
loss_input = tf.placeholder(tf.float32)
tf.summary.scalar("loss",loss_input)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR) 

saver = tf.train.Saver(network.get_vars())
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded for the network:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
else:  
  print("Could not find old checkpoint for the network")
 

total_params = count_number_trainable_params()
print("NUMBER OF PARAMETERS = %d"%total_params)

print("Press Ctrl+C to save the model and stop training.")

def signal_handler(signal, frame):
  global stop_requested
  print('You just pressed Ctrl+C! Saving model.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = training_thread.global_t)
  stop_requested = True
  sys.exit()
  
 
signal.signal(signal.SIGINT, signal_handler)

global_t = training_thread.process(saver,sess, global_t, summary_writer, summary_op, loss_input)
print('Now saving data. Please wait')
  
saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

