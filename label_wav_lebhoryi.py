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

#### update 2020.3.28 #############################
# 1. 将--wav 改为 --dir_path , 获取音频文件夹路径
# 2. 新增识别准确率输出
# 3. 执行命令如下：
#     python3 label_wav \
#     --dir_path=../local_data/test_wav/other \
#     --graph=./pb/327_dnn.pb \
#     --labels=./train_model/326_dnn/dnn_labels.txt
###################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # Only showing warning & Error

import librosa
from python_speech_features import mfcc as pmfcc
import glob
import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
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
              count_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-count_top_predictions:][::-1]
    # for node_id in top_k:
    #   human_string = labels[node_id]
    #   score = predictions[node_id]
    #   print('%s (score = %.5f)' % (human_string, score))

    # 只保留分值最高的标签
    return labels[int(top_k[:1])]


def label_wav(dir_path, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  # wav file
  if not dir_path or not tf.gfile.Exists(dir_path):
    # tf.logging.fatal('Audio file does not exist %s', wav)
    tf.logging.fatal('Audio file path does not exist %s', dir_path)

  # label file
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  # model file
  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  # 原本是wav, 改成了dir_path
  # 完整的音频路径 list
  # 识别错误的样本总数 | 在other时，识别为xrxr的数量 | 在other时，识别为nihaoxr的数量
  count, count1, count2 = 0, 0, 0
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  # 文件夹前四个字母
  dir_name = os.path.basename(dir_path)[:4]
  for wav in wav_paths:
    # wav = os.path.join(os.path.dirname(wav_paths[i]), str(i) + ".wav")
    # os.rename(wav_paths[i], wav)  # rename

    # 获取mfcc
    # y,sr = librosa.load(wav)
    # mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
    # print("{} 的mfcc 为：".format(os.path.basename(wav)))
    # with open("/home/lebhoryi/Desktop/mfcc.txt", "w") as f:
    #     f.write(str(mfcc))

    with open(wav, 'rb') as wav_file:
      wav_data = wav_file.read()

    label = run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

    # print(os.path.split(os.path.dirname(wav)))
    # 打印所有样本的预测信息
    print('{} : {}'.format(os.path.basename(wav), label))

    # 根据文件夹名字来分别xrxr \ nhxr \ other
    if dir_name[:4] == "othe":
      count1 = count1 + 1 if label == "xrxr" else count1
      count2 = count2 + 1 if label == "nihaoxr" else count2
      print('{} : {}'.format(os.path.basename(wav), label))
    elif dir_name[:4] == "xrxr":
      if label != "xrxr":
        count += 1
        print('{} : {}'.format(os.path.basename(wav), label))
    elif dir_name[:4] == "niha":
      if label != "nihaoxr":
        count += 1
        print('{} : {}'.format(os.path.basename(wav), label))
  if dir_name[:4] == "othe":
    count = count1 + count2
    print("总共有{}个样本, 识别错误的有{}条样本".format(len(wav_paths), count))
    print("识别为xrxr：{}, 识别为nihaoxr：{}".format(count1, count2))
  else:
    print("总共有{}个样本, 识别错误的有{}条样本".format(len(wav_paths), count))
  print("准确率为：{}".format(1 - count / len(wav_paths)))

def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.dir_path, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument(
  #     '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--dir_path', type=str, default='../local_data/test_wav/2_classes/other',
      help='Wav files path.')
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
      help='countber of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
