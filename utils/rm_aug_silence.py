# coding=utf-8
'''
@ Summary: 最终的效果是用脚本分离出1s和超过1s的，然后超过1s的手动进行剪辑，
           音频文件名不改，在同一个音频文件夹路径下会生成两个音频文件夹，分别存放
           超过1s和1s的音频

@ file:    rm_aug_silence.py
@ version: 1.0.0

@ Update:  增加pathlib.Path() 这个库，可以无视平台差异
@ Version: 1.0.1

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/3/26 下午4:41
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import shutil
from pathlib import Path
from pydub import AudioSegment


def detect_leading_silence(sound, silence_threshold=-35.0, chunk_size=20):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop

    # for i in range(1 , len(sound)+1, chunk_size):
    #     print(sound[i:i+chunk_size].dBFS)
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def remove_aug_silence2(dir_path):
    if not os.path.exists(dir_path):
        raise Exception("No " + dir_path + "!")

    # x, y, z, w = 0, 0, 0, 0  # 统计超过1s语音的个数
    x = 1

    # 新的剪辑之后，1s长度的语音存放的文件夹
    new_path = os.path.join(dir_path, "1s")
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    # 长度超过一秒需要手动剪辑的语音存放路径
    long_path = os.path.join(dir_path, "long")
    if not os.path.isdir(long_path):
        os.mkdir(long_path)

    # 获取所有的.wav 文件路径列表
    # 格式['0.wav', '1.wav', ...]
    wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
    for i in range(len(wav_files)):
        # 读取文件
        sound = AudioSegment.from_file(wav_files[i], format="wav")

        # 减去了两个数值是为了增加前后的静音区
        start_trim = detect_leading_silence(sound, -40)
        # start_trim 不能为负，否则会生成空白的语音
        start_trim = start_trim - 50 if start_trim >= 50 else start_trim
        end_trim = detect_leading_silence(sound.reverse(), -40)
        end_trim = end_trim - 100 if end_trim >= 100 else end_trim

        # durtion 单位 ms 1s=1000ms
        duration = len(sound) - end_trim - start_trim

        # 储存的wav文件名字
        # file_name = os.path.basename(wav_files[i])
        if int(x) < 10:
            x = "00" + str(x)
        elif int(x) < 100:
            x = "0" + str(x)
        else:
            x = str(x)
        file_name = "001" + x + ".wav"
        x = int(x) + 1
        # 如果剪了  头尾静音区之后的语音时长小于1s,时长限定为1s
        if duration <= 1000:
            new_sound = sound[start_trim: start_trim+1000]
            new_sound.export(os.path.join(new_path, file_name), format="wav")
        elif duration <= 1050:
            start_trim2 = start_trim - 25 if start_trim >= 25 else start_trim
            new_sound2 = sound[start_trim2: start_trim2+1000]
            new_sound2.export(os.path.join(new_path, file_name), format="wav")
        else:    # 大于1s的, 需要手动剪辑
            newsound = sound[start_trim: len(sound)-end_trim]
            newsound.export(os.path.join(long_path, file_name), format="wav")
            print("{} 的时长为： {}s...".format(file_name, duration/1000))
        # print("正在剪辑第{}条语音...".format(i))
    # print("有{}条语音小于1050ms...".format(x))  # 20
    # print("有{}条语音小于1100ms...".format(y))  # 23
    # print("有{}条语音小于1150ms...".format(z))  # 9
    # print("有{}条语音大于1150ms...".format(w))  # 25

def remove_wav(wav, wav_1s, wav_long):
    """ 单个音频剪掉静音区 """
    assert wav, print("No audio file exists!")

    if not wav_1s.exists():  wav_1s.mkdir()

    # 读取文件
    sound = AudioSegment.from_file(wav, format="wav")

    # 减去了两个数值是为了增加前后的静音区 -35
    start_trim = detect_leading_silence(sound, -30)
    # start_trim 不能为负，否则会生成空白的语音
    start_trim = start_trim - 50 if start_trim >= 50 else start_trim
    end_trim = detect_leading_silence(sound.reverse(), -30)
    end_trim = end_trim - 100 if end_trim >= 100 else end_trim

    # durtion 单位 ms 1s=1000ms
    duration = len(sound) - end_trim - start_trim

    # 如果剪了头尾静音区之后的语音时长小于1s,时长限定为1s
    start_trim2 = len(sound) - end_trim - 1000
    if start_trim2 < 0:
        start_trim2 = 0
    if start_trim > 400:
        start_trim2 = start_trim

    if duration <= 1050:
        new_sound = sound[start_trim2: start_trim2+1000]
        new_sound.export(wav_1s/wav.name, format="wav")
        print(f"{wav.name} 1s 音频剪辑成功...")
    # elif duration <= 1050:
    #     start_trim2 = start_trim - 25 if start_trim >= 25 else start_trim
    #     new_sound2 = sound[start_trim2: start_trim2+1000]
    #     new_sound2.export(wav_1s/wav.name, format="wav")
    #     print(f"{wav.name} 1s 音频剪辑成功...")
    else:    # 大于1s的, 需要手动剪辑
        newsound = sound[start_trim: len(sound)-end_trim]
        # newsound = sound[start_trim: start_trim+1000]
        newsound.export(wav_long/wav.name, format="wav")
        print("{} 的时长为： {}s...".format(wav.name, duration/1000))


def remove_aug_silence(dir_path):

    assert dir_path.exists(), Exception("No " + str(dir_path) + "!")

    # 新的剪辑之后，1s长度的语音存放的文件夹
    new_path = dir_path / "1s"
    if not new_path.exists():  new_path.mkdir()

    # 长度超过一秒需要手动剪辑的语音存放路径
    long_path = dir_path / "long"
    if not long_path.exists(): long_path.mkdir()

    # 获取所有的.wav 文件路径列表
    wav_files = dir_path.glob('*.wav')
    for wav in wav_files:
        remove_wav(wav, new_path, long_path)
    print("剪辑完成, 剩下的需要手工剪辑啦...")


def merge_wavs(root, new_path):
    """ 将所有的音频整合到一个文件夹中 """
    assert root.exists(), Exception("No files path exists!")
    i = 0

    if not new_path.exists():  new_path.mkdir()

    for dir in root.iterdir():
        print(dir)
        wav_paths = dir.glob('*.wav')
        for wav in wav_paths:
            i += 1
            shutil.copy(wav, new_path / (str(i)+'.wav'))
        print(i)
    print(f"共有{len(list(new_path.iterdir()))}条音频文件...")
    return new_path


if __name__ == "__main__":
    root_path = "../../local_data/web_data_train/20200722/long"
    root_path = Path(root_path)
    wavs_path = root_path.parent / 'aug_xrxr'

    remove_aug_silence(root_path)

    # 合并所有音频文件
    # wavs_path = merge_wavs(root_path, wavs_path)

    # 单个音频
    # file_path = '/home/lebhoryi/RT-Thread/WakeUp-Xiaorui/local_data/' \
    #             '328_data/audio2/60.wav'
    # tmp = Path('/home/lebhoryi/RT-Thread/WakeUp-Xiaorui/local_data/'
    #            '328_data/tmp')
    # file_path = Path(file_path)
    # remove_wav(file_path, tmp, tmp)


