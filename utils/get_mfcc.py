# coding=utf-8
'''
@ Summary: 提取训练音频的mfcc, 代码都是从input_data.py 抄的，就没加很多注释
@ Update:  

@ file:    get_mfcc.py
@ version: 1.0.0

@ Update:  simplify how to getting mfcc
@ Version: 1.0.2

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/3/30 下午12:05
'''

import os.path
import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

def get_mfcc(time_shift_padding, time_shift_offset, background_reshaped, background_volume):
    ##### get mfcc ##########
    wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)

    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
        scaled_foreground,
        time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 time_shift_offset_placeholder_,
                                 [desired_samples, -1])
    # Mix in background noise.
    background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    background_volume_placeholder_ = tf.placeholder(tf.float32, [])

    background_mul = tf.multiply(background_data_placeholder_,
                                 background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)


    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True)
    mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=dct_coefficient_count)

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    input_dict = {
        wav_filename_placeholder_: "../../data/xrxr/1.wav",
        foreground_volume_placeholder_: 1,
        time_shift_padding_placeholder_: time_shift_padding,
        time_shift_offset_placeholder_: time_shift_offset,
        background_data_placeholder_: background_reshaped,
        background_volume_placeholder_: background_volume,

    }
    _mfcc = sess.run(mfcc_, feed_dict=input_dict)
    return _mfcc

def get_time_shift(time_shift_ms, sample_rate):
    ##### get time_shift_padding & time_shift_offset #############
    # time_shift = int((time_shift_ms * sample_rate) / 1000)
    time_shift = 0
    # If we're time shifting, set up the offset for this sample.
    if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
    else:
        time_shift_amount = 0
    if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
    else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]

    return time_shift_padding, time_shift_offset

def prepare_background_data(BACKGROUND_NOISE_DIR_NAME):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """

    data_dir = "../data"
    background_data = []
    background_dir = os.path.join(data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
        return background_data
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        search_path = os.path.join(data_dir, BACKGROUND_NOISE_DIR_NAME,
                                   '*.wav')
        for wav_path in gfile.Glob(search_path):
            wav_data = sess.run(
                wav_decoder,
                feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
            background_data.append(wav_data)
    return background_data

def get_back(desired_samples, background_volume_range, background_data):
    #### get background_data_placeholder_ ################

    # Choose a section of background noise to mix in.
    background_index = np.random.randint(len(background_data))
    background_samples = background_data[background_index]
    background_offset = np.random.randint(
        0, len(background_samples) - desired_samples)
    background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
    background_reshaped = background_clipped.reshape([desired_samples, 1])
    if np.random.uniform(0, 1) < background_frequency:
        background_volume = np.random.uniform(0, background_volume_range)
    else:
        background_volume = 0
    return background_reshaped, background_volume


def get_mfcc_simplify(wav_filename, desired_samples,
                      window_size_samples, window_stride_samples):
    wav_loader = io_ops.read_file(wav_filename)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)

    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True)

    print(spectrogram)
    mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=dct_coefficient_count)

    # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    # mfcc_ = sess.run(mfcc_)

    return mfcc_


if __name__ == "__main__":
    sample_rate, window_size_ms, window_stride_ms = 16000, 40, 20
    dct_coefficient_count = 30
    clip_duration_ms = 1000
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    _mfcc = get_mfcc_simplify("../../data/xrxr/1.wav", desired_samples,
                              window_size_samples, window_stride_samples)

    print(_mfcc)

    # time_shift_ms = 100
    # background_frequency = 0.8
    # background_volume_range = 0.1
    # BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

    # time_shift_padding, time_shift_offset = get_time_shift(time_shift_ms, sample_rate)
    # # background_data = prepare_background_data(BACKGROUND_NOISE_DIR_NAME)
    # # background_reshaped, background_volume = \
    # #     get_back(desired_samples, background_volume_range, background_data)
    # background_reshaped = np.zeros([desired_samples, 1])
    # background_volume = 0
    #
    # _mfcc = get_mfcc(time_shift_padding, time_shift_offset, background_reshaped, background_volume)
    # print(_mfcc)
    # print(np.array(_mfcc, dtype=np.int8))


    # import librosa
    #
    # 获取mfcc
    # wav = "../data/nihaoxr/1.wav"
    # y,sr = librosa.load(wav)
    # # print(sr)
    # mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40,)
    # print("{} 的mfcc 为：\n{}".format(os.path.basename(wav), mfcc))

    # from python_speech_features import mfcc
    # from scipy.io import wavfile
    # (rate,sig) = wavfile.read(wav)
    # mfcc2 = mfcc(sig, samplerate=16000,
    #         winlen=window_size_ms/1000, winstep=window_stride_ms/1000, numcep=13,
    #         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
    #         ceplifter=22, appendEnergy=True)
    # print(mfcc2)

