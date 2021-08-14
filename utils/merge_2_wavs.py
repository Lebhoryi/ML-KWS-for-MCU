# coding=utf-8
'''
@ Summary: 合并两个音频文件
@ Update:  

@ file:    merge_2_wavs.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/8/17 下午3:48
'''
import wave
import numpy as np
import pyaudio
from pathlib import Path

def merge_wavs(file1, file2):
    f1 = wave.open(file1, 'rb')
    f2 = wave.open(file2, 'rb')

    # 音频1的数据
    params1 = f1.getparams()
    nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1 = params1[:6]
    print(nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1)
    f1_str_data = f1.readframes(nframes1)
    f1.close()
    f1_wave_data = np.fromstring(f1_str_data, dtype=np.int16)

    # 音频2的数据
    params2 = f2.getparams()
    nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2 = params2[:6]
    print(nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2)
    f2_str_data = f2.readframes(nframes2)
    f2.close()
    f2_wave_data = np.fromstring(f2_str_data, dtype=np.int16)

    # 对不同长度的音频用数据零对齐补位
    if nframes1 < nframes2:
        length = abs(nframes2 - nframes1)
        temp_array = np.zeros(length, dtype=np.int16)
        rf1_wave_data = np.concatenate((f1_wave_data, temp_array))
        rf2_wave_data = f2_wave_data
    elif nframes1 > nframes2:
        length = abs(nframes2 - nframes1)
        temp_array = np.zeros(length, dtype=np.int16)
        rf2_wave_data = np.concatenate((f2_wave_data, temp_array))
        rf1_wave_data = f1_wave_data
    else:
        rf1_wave_data = f1_wave_data
        rf2_wave_data = f2_wave_data

    # ================================
    # 合并1和2的数据
    new_wave_data = rf1_wave_data + rf2_wave_data
    new_wave = new_wave_data.tostring()
    return new_wave


# 实现录音
def record(re_frames, WAVE_OUTPUT_FILENAME):
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(re_frames)
    wf.close()


if __name__ == "__main__":
    p = pyaudio.PyAudio()
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    RATE = 16000
    root_path = Path("/home/lebhoryi/Tmp")
    noise_dir = root_path / 'noise'
    noise_list = noise_dir.glob("*.wav")
    real_wav_dir = root_path / 'xrxr'
    real_wav_list = real_wav_dir.glob('*.wav')

    for file2 in real_wav_list:
        for file1 in noise_dir.glob("*.wav"):
            new_wav = merge_wavs(str(file1), str(file2))
            record(new_wav, str(root_path / 'xrxr_20200817' / (file1.stem + '_' + file2.name)))
    print('Done~')