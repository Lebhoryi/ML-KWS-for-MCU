# coding=utf-8
'''
@ Summary: rename files in ../../local_data/test_wav
@ Update:  

@ file:    rename_files.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/5/27 上午11:01
'''

from pathlib import Path

def rename(dir):
    wavs = list(dir.glob('*.wav'))
    for i in range(len(wavs)):
        wav = wavs[i]
        wav.rename(wav.parent / (str(i) + '.wav'))

if __name__ == '__main__':
    wav_path = '../../local_data/test_wav'
    wav_path = Path(wav_path)
    # for dir in wav_path.iterdir():
    #     rename(dir)

    wavs = list(wav_path.glob('*/*.wav'))
    print(f"总共有{len(wavs)} 个文件")