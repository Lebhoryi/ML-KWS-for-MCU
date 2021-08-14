# coding=utf-8
'''
@ Summary: 仅在tensorlfow2 下面运行
           tf1.14 两行代码实现mfcc提取，现用tf2 分布实现 提取mfcc
@ Update:  tf2 官网例程 梅尔滤波计算过程中少了一步对spectrograms 的根号


@ file:    tf2_mfccs.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/5/9 下午5:22
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import io_ops
from tensorflow import audio


def load_wav(wav_path, sample_rate=16000):
    '''
        load one wav file

    Args:
        wav_path: the wav file path, str
        sample_rate: wav's sample rate, int8

    Returns:
        wav: wav文件信息, 有经过归一化操作, float32
        rate: wav's sample rate, int8

    '''
    wav_loader = io_ops.read_file(wav_path)
    (wav, rate) = audio.decode_wav(wav_loader,
                                   desired_channels=1,
                                   desired_samples=sample_rate)
    # shape (16000,)
    wav = np.array(wav).flatten()
    return wav, rate


def stft(wav, win_length=640, win_step=640, n_fft=1024):
    '''
        stft 快速傅里叶变换

    Args:
        wav: *.wav的文件信息, float32, shape (16000,)
        win_length: 每一帧窗口的样本点数, int8
        win_step: 帧移的样本点数, int8
        n_fft: fft 系数, int8

    Returns:
        spectrograms: 快速傅里叶变换计算之后的语谱图
                shape: (1 + (wav-win_length)/win_step, n_fft//2 + 1)
        num_spectrogram_bins: spectrograms[-1], int8

    '''
    # if fft_length not given
    # fft_length = 2**N for integer N such that 2**N >= frame_length.
    # shape (25, 513)
    stfts = tf.signal.stft(wav, frame_length=win_length,
                           frame_step=win_step, fft_length=n_fft)
    spectrograms = tf.abs(stfts)

    spectrograms = tf.square(spectrograms)


    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape.as_list()[-1]  # 513
    return spectrograms, num_spectrogram_bins


def build_mel(spectrograms, num_mel_bins, num_spectrogram_bins,
              sample_rate, lower_edge_hertz, upper_edge_hertz):
    '''
        构建梅尔滤波器

    Args:
        spectrograms: 语谱图 (1 + (wav-win_length)/win_step, n_fft//2 + 1)
        num_mel_bins: How many bands in the resulting mel spectrum.
        num_spectrogram_bins：
            An integer `Tensor`. How many bins there are in the
            source spectrogram data, which is understood to be `fft_size // 2 + 1`,
            i.e. the spectrogram only contains the nonredundant FFT bins.
            sample_rate: An integer or float `Tensor`. Samples per second of the input
            signal used to create the spectrogram. Used to figure out the frequencies
            corresponding to each spectrogram bin, which dictates how they are mapped
            into the mel scale.
        sample_rate: 采样率
        lower_edge_hertz:
            Python float. Lower bound on the frequencies to be
            included in the mel spectrum. This corresponds to the lower edge of the
            lowest triangular band.
        upper_edge_hertz:梅尔滤波器的最高频率，梅尔滤波器的最高频率



    Returns:
        mel_spectrograms: 梅尔滤波器与语谱图做矩阵相乘之后的语谱图
                shape: (1 + (wav-win_length)/win_step, n_mels)

    '''
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz)
    # tf.print('linear_to_mel_weight_matrix : {}'.format(
    #     tf.transpose(linear_to_mel_weight_matrix, [1,0])[0]))

    tf.print(spectrograms.shape)
    tf.print(linear_to_mel_weight_matrix.shape)

    ################ 官网教程中, 少了sqrt #############
    spectrograms = tf.sqrt(spectrograms)
    mel_spectrograms = tf.tensordot(spectrograms,
                        linear_to_mel_weight_matrix, 1)

    # 两条等价
    # mel_spectrograms = tf.matmul(spectrograms, linear_to_mel_weight_matrix)

    # shape (25, 40)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    return mel_spectrograms


def log(mel_spectrograms):
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    # shape: (1 + (wav-win_length)/win_step, n_mels)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-12)
    return log_mel_spectrograms


def dct(log_mel_spectrograms, dct_counts):
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    # shape (1 + (wav-win_length)/win_step, dct)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)
    # 取低频维度上的部分值输出，语音能量大多集中在低频域，数值一般取13。
    mfcc = mfccs[..., :dct_counts]
    return mfcc


if __name__ == '__main__':
    path = '/home/lebhoryi/Music/0.wav'
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20, 4000, 40
    n_fft = 1024
    win_length, win_step = 640, 640
    dct_counts = 10

    wav, rate = load_wav(path)
    # tf.print('wav : {}'.format(wav))

    spec, num_spec_bins = stft(wav, win_length=640, win_step=640, n_fft=n_fft)

    mel_spectrograms = build_mel(spec, num_mel_bins=num_mel_bins,
                                 num_spectrogram_bins=num_spec_bins,
                                 sample_rate=rate,
                                 lower_edge_hertz=lower_edge_hertz,
                                 upper_edge_hertz=upper_edge_hertz)


    log_mel_spectrograms = log(mel_spectrograms)

    mfccs = dct(log_mel_spectrograms, dct_counts)

    # mfccs_2 = mfccs * 2
    # mfccs_2 = mfccs_2.numpy()
    # mfccs_2 = mfccs_2.flatten()
    # tf.print(mfccs_2[39:45])

