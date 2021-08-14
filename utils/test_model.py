# coding=utf-8
'''
@ Summary: copy from ../label_wav_lebhoryi.py
@ Update:

@ file:    test_model.py
@ version: 1.0.0

@ Update:  增加了三分类的预测
@ Date:    2020/07/23

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/3/31 下午2:12
'''

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


import librosa
from python_speech_features import mfcc as pmfcc
import glob
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score, precision_score, \
            recall_score, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.python.platform import gfile


os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # Only showing warning & Error
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


def run_graph(dir_name, wav_data, labels, input_layer_name, output_layer_name,
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

        d = {}  # 对应的标签和概率
        if labels[top_k[0]] == dir_name:
            print(f"true label: {dir_name}, score: {predictions[top_k[0]]*100:.2f}")
        elif labels[top_k[0]] != dir_name:
            print(f"wrong label: {labels[top_k[0]]}, true label: {dir_name}\n,\
                  score: {predictions[top_k[0]]*100:.2f}")
        for node_id in top_k:
          human_string = labels[node_id]
          score = predictions[node_id]
          d[human_string] = score
        #   print('%s (score = %.5f)' % (human_string, score))
        # 只保留分值最高的标签 和 对应的标签概率 dict
        return labels[int(top_k[:1])], d


def label_wav(dir_path, labels, graph, input_name, output_name, how_many_labels, average,
              how_many_classes):
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

    dir_lenth = 0
    # 获取有多少个文件夹,创建混淆矩阵用
    for _, paths, _ in os.walk(dir_path):
        for _ in paths:
            dir_lenth += 1

    # 完整的音频路径 list
    search_path = os.path.join(dir_path, '*', '*.wav')
    wav_paths = gfile.Glob(search_path)
    print('data_size = {}'.format(len(wav_paths)))

    # 真标签和预测标签
    y_true, y_pred, y_scores = [0] * len(wav_paths), [0] * len(wav_paths), \
                            [0] * len(wav_paths)
    for i in range(len(wav_paths)):
        wav = wav_paths[i]
        dir_name = Path(wav).parent.name

        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()

        label, d = run_graph(dir_name, wav_data, labels_list, input_name, output_name, how_many_labels)

        # 打印预测的值
        dir_name = "".join(os.path.split(os.path.dirname(wav))[-1:])[:4]
        # 打印所有样本的预测信息
        # print('{} : {}'.format(os.path.basename(wav), label))


        if how_many_classes == 2:
            y_true[i] = 1 if dir_name == "xrxr" else 0
            y_pred[i] = 1 if label == "xrxr" else 0
        else:
            if dir_name == "niha":
                y_true[i] = 1
            elif dir_name == "xrxr":
                y_true[i] = 2
            else:
                y_true[i] = 3
            if label == "nihaoxr":
                y_pred[i] = 1
            elif label == "xrxr":
                y_pred[i] = 2
            else:
                y_pred[i] = 3

    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score( y_true, y_pred, average=average)

    # plt.figure(1) # 创建图表1
    # plt.title('Precision/Recall Curve')# give plot a title
    # plt.xlabel('Recall')# make axis labels
    # plt.ylabel('Precision')

    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # plt.figure(1)
    # plt.plot(precision, recall)
    # plt.show()
    # plt.savefig('p-r.png')

    print("Confusion Matrix is:\n{}".format(matrix))
    print("The accuracy is {:.2f}%.".format(acc * 100))
    print("The precision is {:.2f}%.".format(p * 100))
    print("The recall is {:.2f}%.".format(r * 100))
    print("The f1 scores is {:.2f}%.".format(f1 * 100))


def main(_):
    """Entry point for script, converts flags to arguments."""
    label_wav(FLAGS.dir_path, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
              FLAGS.output_name, FLAGS.how_many_labels, FLAGS.average,
              FLAGS.how_many_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--wav', type=str, default='', help='Audio file to be identified.')
    parser.add_argument(
        '--dir_path', type=str, default='../../tmp',
        help='Wav files path.')
    parser.add_argument(
        '--graph', type=str, default='', help='Model to use for identification.')
    parser.add_argument(
        '--average', type=str, default='macro', help='The mode to caculate precision recall.')
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
        default=14,
        help='countber of results to show.')
    parser.add_argument(
        '--how_many_classes', type=int, default=3, help="The number of classes."
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
