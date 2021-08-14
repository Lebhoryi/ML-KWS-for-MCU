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
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""


############################################
# update: 2020/05/23
#         input convert raw audio to mfcc
############################################

############################################
# update: 2020/07/28
#         麦克风获取音频流进行测试
############################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import pyaudio
import wave
import numpy as np
import librosa
from queue import Queue
from pathlib import Path
import time


# pylint: disable=unused-import
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions, FLAG):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    index = predictions.argmax()
    top = predictions.argsort()
    # print(labels[index])
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    if labels[top_k[0]] == "xrxr" and predictions[top_k[0]]>= 0.9:
      if FLAG == 14:
        print('%s (score = %.5f)' % (labels[top_k[0]], predictions[top_k[0]]))
        print("=="*15)
        print()
      FLAG += 1
      return FLAG
    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  ########################
  # update 2020/07/23
  # audio stream input

  CHUNK = 320  # 20 ms 帧移
  CHANNELS = 1  # 单通道
  RATE = 16000  # 16k 采样率
  FRAMES = 49  # 49 帧
  FORMAT = pyaudio.paInt16
  HEAD = b'RIFF$}\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80' \
         b'>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00}\x00\x00'

  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
  # print(p.get_sample_size(FORMAT))

  q = Queue()
  # init queue
  for _ in range(50):
      q.put(b"\x00" * CHUNK * 2)

  flag, wav_data = 0, 0
  root_path = Path('../local_data/record')
  if not root_path.exists():  root_path.mkdir()


  print("Start recording...")

  while True:
    if flag == 15:
      # save wav
      i = time.strftime("%m%d%H%M%S", time.localtime())
      wav_path = root_path / (str(i) + '.wav')
      with wav_path.open('wb') as wf:
        wf = open(str(wav_path), 'wb')
        wf.write(wav_data)
      # init queue
      for _ in range(50):
        q.get()
        q.put(b"\x00" * CHUNK * 2)
      flag = 0
      sys.stdout.flush()
      continue
    data = stream.read(CHUNK)
    # 入队
    q.put(data)
    # 出队
    q.get()

    wav_data = HEAD + b''.join(list(q.queue))
    flag = run_graph(wav_data, labels_list, input_name,
                     output_name, how_many_labels, flag)


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
