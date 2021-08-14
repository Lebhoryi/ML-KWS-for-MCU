# coding=utf-8
'''
@ Summary: 获取每一层的输出，对官方代码进行动刀


@ file:    quant_test.py
@ version: 1.0.0


@ Author:  Lebhoryi@gmail.com
@ Date:    2020/4/28 15:01
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import sys
import numpy as np
import tensorflow as tf

import models_official as model

path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../'))

import input_data

def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info, every_network_output):
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
    model_size_info: Model dimensions : different lengths for different models
  """

  # 输出tf的日志,有五个级别,DEBUG，INFO，WARN，ERROR和FATAL,默认是WARN,此处设为INFO
  tf.logging.set_verbosity(tf.logging.INFO)

  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = model.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)


  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)

  label_count = model_settings['label_count']
  fingerprint_size = model_settings['fingerprint_size']

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  # get loss
  # if model_architecture == "dnn" or model_architecture == "ds_cnn":
  if model_architecture == "dnn":
    # with act_max
    outputs = models.create_model(
            fingerprint_input,
            model_settings,
            FLAGS.model_architecture,
            FLAGS.model_size_info,
            FLAGS.act_max,
            is_training=False)
  else:
    # without act_max
    outputs = model.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=False)

  logits = outputs[-1]
  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  predicted_indices = tf.argmax(logits, 1)  # 返回的是logits中的最大值的索引号
  expected_indices = tf.argmax(ground_truth_input, 1)
  # 对比这两个矩阵或者向量的相等的元素
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  # 得到混淆矩阵
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  model.load_variables_from_checkpoint(sess, FLAGS.checkpoint)
  # tmp = sess.run(outputs[0])

  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None

  # 每一个样本的每一层输出的最大和最小值
  all_max_output, all_min_output = [[] for _ in range(len(outputs))], \
                                   [[] for _ in range(len(outputs))]
  # 保存每一层输出的最大值和最小值
  max_output, min_output = [0] * (len(outputs)+1), [0] * (len(outputs)+1)
  for i in range(0, set_size, FLAGS.batch_size):
      test_fingerprints, test_ground_truth = audio_processor.get_data(
         FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)

      # update 2020/05/28
      # 获取每一层的输出的值域
      layers_outputs = sess.run(outputs,
          feed_dict={fingerprint_input: test_fingerprints})
      max_output[0] = max(max_output[0], test_fingerprints.max())
      min_output[0] = min(min_output[0], test_fingerprints.min())
      for index, l_output in enumerate(layers_outputs):
          max_output[index+1] = max(max_output[index+1], l_output.max())
          min_output[index+1] = min(min_output[index+1], l_output.min())
          for single_output in l_output:
              all_max_output[index].append(single_output.max())
              all_min_output[index].append(single_output.min())

      test_accuracy, conf_matrix = sess.run(
          [evaluation_step, confusion_matrix],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
          })
      batch_size = min(FLAGS.batch_size, set_size - i)
      total_accuracy += (test_accuracy * batch_size) / set_size
      if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
      else:
          total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                     set_size))

  # print(os.path.isdir(every_network_output))
  # with open(every_network_output, 'a') as f:
  #     f.write('Test accuracy = %.2f%% (N=%d)\n' % (total_accuracy * 100,
  #                                                set_size))
  for i in range(len(max_output)):
      if i == 0:
          tf.logging.info(f'input 的最大值是{max_output[i]}, 最小值是{min_output[i]}')
          with open(every_network_output, 'a') as f:
              f.write(f'input 输出的最大值是：{max_output[i]},'
                      f' 最小值是：{min_output[i]}' + '\n')
      else:
          tf.logging.info(f'第{i}层的最大值是{max_output[i]}, 最小值是{min_output[i]}')
          with open(every_network_output, 'a') as f:
              f.write(f'第{i}层输出的最大值是：{max_output[i]},'
                      f' 最小值是：{min_output[i]}' + '\n')


def main(_):

  # Create the model, load weights from checkpoint and run on train/val/test
  run_quant_inference(FLAGS.wanted_words, FLAGS.sample_rate,
      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
      FLAGS.model_architecture, FLAGS.model_size_info, FLAGS.output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      # default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      default=" ",
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../data',
      help="""\
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
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
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
      default='',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--act_max',
      type=float,
      nargs="+",
      default=[32,0,0,0,0,0,0,0,0,0,0,0],
      help='activations max')
  parser.add_argument(
      '--output',
      type=str,
      default='../weights_h/606_cnn_with_2x_lr/every_network_output.txt',
      help='The name of weight dir.')


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
