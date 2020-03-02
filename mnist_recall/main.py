# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from model import MNISTRecallModel
from training_thread import TrainingThread
from rmsprop_applier import RMSPropApplier

from constants import MAX_TIME_STEP
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import INITIAL_ALPHA
from constants import CHECKPOINT_DIR

import pdb
import sys


device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = INITIAL_ALPHA

global_t = 0

stop_requested = False

network = MNISTRecallModel(-1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

training_thread = TrainingThread(network,initial_learning_rate,learning_rate_input,grad_applier,MAX_TIME_STEP,device)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
loss_input = tf.placeholder(tf.float32)
tf.summary.scalar("loss",loss_input)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
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
 
def signal_handler(signal, frame):
  global stop_requested
  print('You just pressed Ctrl+C! Saving model.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = training_thread.global_t)
  stop_requested = True
  sys.exit()
  
signal.signal(signal.SIGINT, signal_handler)

print('Press Ctrl+C to save the model and stop training.')

global_t = training_thread.process(saver, sess, global_t, summary_writer, summary_op, loss_input)

print('Now saving data. Please wait')
  
saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

