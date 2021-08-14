# coding=utf-8
'''
@ Summary: 获取wav音频数据
@ Update:  1.0.2 计算wav的mfcc数据


@ file:    get_output_from_network.py
@ version: 2.0.0 获取cnn 网络的中间变量并输出


@ version: 2.0.1 代码重构

@ version: 2.0.2 保存每一个层输出的最大值和最小值

@ Date:    2020/05/27
           需要对批量数据推理时的每一层的输出；
           CNN好像出了点玄学问题，转到office_get_layers_output.py 继续更新


@ Author:  Lebhoryi@gmail.com
@ Date:    2020/4/28 15:01
'''

import os
import sys


import argparse
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from models_rebuild import *


os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  # Only showing warning & Error
sys.path.append("..")

def load_files(path):
    # load trained variables
    f = open(path)
    lines = f.readlines()
    lines = list(map(lambda x:x[:-1], lines))


    result = list(filter(lambda i: i[:2] != "bn", lines))
    res = []
    return result


def get_mfcc(wav_path):
    '''读取wav, 计算mfcc
    '''
    # get wav
    wav_loader = io_ops.read_file(wav_path)
    # wav_decoder: (audio, sample_rate) (16000, 1)
    wav_decoder = contrib_audio.decode_wav(wav_loader,
                                           desired_channels=1,
                                           desired_samples=16000)

    # stft , get spectrogram
    spectrogram = contrib_audio.audio_spectrogram(
                            wav_decoder.audio,
                            window_size=640,
                            stride=640,
                            magnitude_squared=True)

    # get mfcc (C, H, W)
    _mfcc = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        upper_frequency_limit=4000,
        lower_frequency_limit=20,
        filterbank_channel_count=40,
        dct_coefficient_count=10)

    # mfcc = _mfcc.eval()
    return _mfcc


def main(_):
    # recreate the model, load weights from weight.h and run on test

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = prepare_model_settings(
        FLAGS.label_count,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.dct_coefficient_count)


    # load variable names
    variable_names = load_files(path=FLAGS.variables_path)


    # load variables from weight file
    variable_lines = load_files(path=FLAGS.weight_path)


    # get mfcc from wav file
    mfcc = get_mfcc(FLAGS.wav_path) # (1, 25, 10)
    # shape (B, fingerprint_input)
    input_data = tf.reshape(mfcc, (1, model_settings['fingerprint_size']))


    # test model
    output = create_model(input_data, variable_lines, model_settings,
                        variable_names, FLAGS.label_count,
                        FLAGS.model_architecture,
                        FLAGS.model_size_info)


    sess.run(tf.global_variables_initializer())
    res = sess.run(output)

    # res = tf.reshape(res[-1], (FLAGS.label_count))
    predictions = sess.run(tf.nn.softmax(res[-1]))

    # print(predictions.sum())
    for i in range(len(res)):
        print(f'第{i+1}层输出的最大值是：{res[i].max()},'
              f' 最小值是：{res[i].min()}' + '\n')

        # with open(FLAGS.every_network_output, 'a') as f:
        #     f.write(f'第{i+1}层输出的最大值是：{res[i].max()},'
        #             f' 最小值是：{res[i].min()}' + '\n')
    index = predictions.argmax()
    print(f"predictions : {predictions}")
    print(f"index : {index}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path',               type = str,
            default = '../../data/nihaoxr/2.wav',   help = 'Where to load wav file.')

    parser.add_argument('--weight_path',            type = str,
            default = '../weights_h/521_cnn/without_quant.h',
            help = 'Where to load weight file. ')

    parser.add_argument('--variables_path',         type = str,
            default = '../weights_h/521_cnn/name.txt',
            help = 'The file saved variables name')

    parser.add_argument('--every_network_output',   type = str,
            default = '', required = True,          help = 'Where to save network max/min outputs.')

    parser.add_argument('--dct_coefficient_count',  type = int,
            default = 10,                           help = 'How many bins to use for the MFCC fingerprint',)

    parser.add_argument('--window_size_ms',         type=float,
            default = 40.0,                         help = 'How long each spectrogram timeslice is',)

    parser.add_argument('--window_stride_ms',       type=float,
            default = 40.0,                         help = 'How long each spectrogram timeslice is',)

    parser.add_argument('--sample_rate',            type = int,
            default = 16000,                        help = 'Expected sample rate of the wavs',)

    parser.add_argument('--clip_duration_ms',       type=int,
            default=1000,                           help='Expected duration in milliseconds of the wavs',)

    parser.add_argument('--label_count',            type = int,
            default = 14,                           help='How many labels',)

    parser.add_argument('--model_architecture',     type = str,
            default = 'cnn',                        help = 'What model architecture to use')

    parser.add_argument('--model_size_info',        type = int, nargs = "+",
            default = [28,10,4,1,1,30,10,4,2,1,16,128],
            # default = [128, 128, 128],
            help = 'Model dimensions - different for various models')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

