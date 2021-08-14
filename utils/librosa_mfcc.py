# coding=utf-8
'''
@ Summary: 使用libarosa 获取音频的mfcc
@ Html:  https://zhuanlan.zhihu.com/p/94439062

@ file:    librosa_mfcc.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/5/7 下午3:15
'''

import os
import librosa
import scipy
import numpy as np


wav_path = "../../data/nihaoxr/2.wav"
n_fft, hop_length, n_mfcc = 640, 640, 10
win_length = 640

##### 1.源语音信号， shape = wav.length
wav, sr = librosa.load(wav_path, sr=16000)


##### 2.填充及分帧(无预加重处理)，分帧后所有帧的shape = n_ftt * n_frames

# 默认，n_fft 为傅里叶变换维度
y = np.pad(wav, (0, 0), mode='constant')
# hop_length为帧移，librosa中默认取窗长的四分之一
y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)


##### 3.对所有帧进行加窗，shape = n_frames * n_ftt
# shape = n_ftt * n_frames。librosa中window.shape = n_ftt * 1

# 窗长一般等于傅里叶变换维度，短则填充长则截断
fft_window = librosa.filters.get_window('hann', win_length, fftbins=True)
# 不能直接相乘，需要转换一下维度
fft_window = fft_window.reshape((win_length, 1))

# 原信号乘以汉宁窗函数
# y_frames *= 0.5 - 0.5 * np.cos((2 * np.pi * n) / (win_length - 1))
y_frames *= fft_window

####### 4.STFT处理得到spectrum(频谱，实际是多帧的)
# shape = n_frames * (n_ftt // 2 +1)
fft = librosa.core.fft.get_fftlib()
stft_matrix = fft.rfft(y_frames, n=1024, axis=0)


####### 5.取绝对值得到magnitude spectrum/spectrogram(声谱，包含时间维度，即多帧)
# shape = (n_ftt // 2 +1) * n_frames
magnitude_spectrum = np.abs(stft_matrix)    # 承接上一步的STFT


####### 6.取平方得到power spectrum/spectrogram(声谱，包含时间维度，即多帧)
# shape = (n_ftt // 2 +1) * n_frames
power_spectrum = np.square(magnitude_spectrum)



####### 7.构造梅尔滤波器组，shape = n_mels * (n_ftt // 2 +1)
mel_basis = librosa.filters.mel(sr, n_fft=1024, n_mels=40, fmin=20., fmax=4000,
                    htk=True, norm=None, dtype=np.float32)


####### 8.矩阵乘法得到mel_spectrogram，shape = n_mels * n_frames
# [ n_mels ，(n_ftt // 2 +1) ] * [ (n_ftt // 2 +1) ，n_frames ] =
# [ n_mels，n_frames]
power_spectrum = np.sqrt(power_spectrum)
mel_spectrogram = np.dot(mel_basis, power_spectrum)


####### 9.对mel_spectrogram进行log变换，shape = n_mels * n_frames
log_mel_spectrogram = librosa.core.spectrum.power_to_db(mel_spectrogram,
                    ref=1.0, amin=1e-12, top_db=40.0)


####### 10.IFFT变换，实际采用DCT得到MFCC，shape = n_mels * n_frames
# n表示计算维度，需与log_mel_spectrogram.shape[axis]相同, 否则作填充或者截断处理。
# axis=0表示沿着自上而下的方向，分别选取每一行所在同一列的元素进行运算。
mfcc = scipy.fftpack.dct(log_mel_spectrogram, type=2,
            n=None, axis=0, norm=None, overwrite_x=False)


####### 11.取MFCC矩阵的低维(低频)部分，shape = n_mfcc * n_frames
# 取低频维度上的部分值输出，语音能量大多集中在低频域，数值一般取13
mfcc = mfcc[:10]


# print(mfcc.dtype)
# print(np.array(mfcc, dtype=np.int32))
print("{} 的mfcc 为：\n{}".format(os.path.basename(wav_path), mfcc[0]))
