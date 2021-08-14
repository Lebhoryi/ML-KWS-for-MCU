# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017-2018 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to fold the batch norm parameters into preceding layer
# weights and biases
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import tensorflow as tf

path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../'))
import models
import input_data

FLAGS = None

def fold_batch_norm(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture, model_size_info):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
  """
  
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

 
  fingerprint_input = tf.placeholder(
      tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')

  logits = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      FLAGS.model_size_info,
      is_training=False)

  ground_truth_input = tf.placeholder(
      tf.float32, [None, model_settings['label_count']], name='groundtruth_input')

  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)
  saver = tf.train.Saver(tf.global_variables())

  tf.logging.info('Folding batch normalization layer parameters to preceding layer weights/biases')
  #epsilon added to variance to avoid division by zero
  epsilon  = 1e-3 #default epsilon for tf.slim.batch_norm 
  all_variables = [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

  weight_list = ['Variable:0' if i == 0 else 'Variable_'+str(i*2)+':0' for i in range(3)]
  biase_list = ['Variable_'+str(2*i+1)+':0' for i in range(3)]
  #get batch_norm mean
  mean_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    if 'moving_mean' in v.name]
  for i, mean_var in enumerate(mean_variables):
    mean_name = mean_var.name
    mean_values = sess.run(mean_var)
    variance_name = mean_name.replace('moving_mean','moving_variance')
    variance_var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == variance_name][0]
    variance_values = sess.run(variance_var)
    beta_name = mean_name.replace('moving_mean','beta')
    beta_var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == beta_name][0]
    beta_values = sess.run(beta_var)
    bias_name = biase_list[i]
    bias_var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == bias_name][0]
    bias_values = sess.run(bias_var)
    wt_name = weight_list[i]
    wt_var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == wt_name][0]
    wt_values = sess.run(wt_var)

    #Update weights
    tf.logging.info('Updating '+wt_name)
    # 获取带 BN 的每一个维度
    wt_dim = wt_values.shape[-1]
    # 在每一个维度上进行计算
    if i != 2:
        for l in range(wt_values.shape[3]):
            for k in range(wt_values.shape[2]):
                for j in range(wt_values.shape[1]):
                    for x in range(wt_values.shape[0]):
                        # gamma (scale factor) is 1.0
                        wt_values[x][j][k][l] *= 1.0/np.sqrt(variance_values[l]+epsilon)
    else:
        for l in range(wt_values.shape[1]):
            for k in range(wt_values.shape[0]):
                wt_values[k][l] *= 1.0/np.sqrt(variance_values[l]+epsilon)
    wt_values = sess.run(tf.assign(wt_var,wt_values))

    # Update biases
    tf.logging.info('Updating '+bias_name)
    biase_dim = wt_values.shape[-1]
    for l in range(biase_dim):
        bias_values[l] = (1.0*(bias_values[l]-mean_values[l])/np.sqrt(variance_values[l]+epsilon)) \
                         + beta_values[l]
    bias_values = sess.run(tf.assign(bias_var,bias_values))

  #Write fused weights to ckpt file
  tf.logging.info('Saving new checkpoint at '+FLAGS.checkpoint+'_bnfused')
  saver.save(sess, FLAGS.checkpoint+'_bnfused')



def main(_):

  # Create the model and load its weights.
  fold_batch_norm(FLAGS.wanted_words, FLAGS.sample_rate,
                         FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                         FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
                         FLAGS.model_architecture, FLAGS.model_size_info)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='',
      # default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      # default='/tmp/speech_dataset/',
      default='../../data',
      help="""
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=10,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go,nihaoxr,xrxr',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='../train_model/526_cnn/best/cnn_8884.ckpt-13200',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='cnn2',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
