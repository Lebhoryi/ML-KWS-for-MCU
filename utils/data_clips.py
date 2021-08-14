# coding=utf-8
"""
@ Summary: 读取data数据集，生成10%的test_list & 10%的val_list
@ Update:  将随机的分割改为按照文件的hash值进行筛选，
           与训练时候筛选的方式保持一致

@ file:    data_clips.py
@ version: 1.0.2

@ file:    data_clips.py
@ version: 1.0.3
@ Update:  保留wanted words, 进行训练


@ file:    data_clips.py
@ version: 1.0.4
@ Update:  将非标签文件保存进zero文件夹中

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/3/24 下午12:38
"""

"""
1. read the whole files under a certain folder
2. chose 10% files based on hash
3. copy them to a text
"""

import os
import random
import glob
import hashlib
import re
from tensorflow.python.util import compat
from pathlib import Path
import shutil


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def split_data(dir_path):
    # 从文件夹中抽取百分之十为验证集
    # 获取.wav 文件路径列表,list
    print(dir_path)
    wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
    # 随机化
    random.shuffle(wav_paths)

    # 验证、测试各取10%
    num = int(len(wav_paths) * 0.1)
    val_paths = wav_paths[: num]
    test_paths = wav_paths[num: num*2]

    return val_paths, test_paths

def which_set(filename, validation_percentage=10, testing_percentage=10):
    # 按照文件的hash 值 划分训练集、测试集、验证集
    base_name = os.path.basename(filename)
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    # print(percentage_hash)
    if percentage_hash < validation_percentage:
        result = "validation"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "testing"
    else:
        result = "training"
    return result

def save_file(dir_path, val_txt, test_txt, save_path, want_words, flag=True):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # 获取所有的.wav 文件路径列表
    wav_paths, i = glob.glob(os.path.join(dir_path, "*.wav")), 0


    for file in wav_paths:
        ret = which_set(file)
        # 保存的格式：four/d3831f6a_nohash_4.wav
        if ret == "testing":
            with open(test_txt, "a+") as f:
                print("将{}写入test 文件中....".format(os.path.basename(file)))
                f.write(file[11:]+"\n")

            # update 2020/05/25
            # save test file to another dir
            # 1. create dir
            cur_file = Path(file)
            parent = cur_file.parent  # 返回当前路径的父路径
            new_parent = Path(save_path) / parent.name
            new_file, i = new_parent / (str(i)+'.wav'), i+1
            if not new_parent.exists():
                new_parent.mkdir()


            # 2. save files
            shutil.copyfile(cur_file, new_file)

            print(f"copy {cur_file} to {new_parent.name} dir...")


        if flag:
            if ret == "validation":
                with open(val_txt, "a+") as f:
                    print("将{}写入val 文件中....".format(os.path.basename(file)))
                    f.write(file[11:]+"\n")
            else:
                with open(train_txt, "a+") as f:
                    print("将{}写入train 文件中...".format(os.path.basename(file)))
                    f.write(file[11:]+"\n")

def save_other_to_zero(save_path, want_words):
    # 将非标签文件保存进zero文件夹中
    p = Path(save_path)
    new_path = p / '123'
    new_path.mkdir()
    count = 0
    all_others = list(p.glob('*/*.wav'))
    for i in range(len(all_others)):
        # print(all_others[i].parent)
        if all_others[i].parent.name not in want_words:
            new_file, count = new_path / (str(count) + '.wav'), count + 1
            shutil.copyfile(all_others[i], new_file)


def modif_dir(save_path, want_words):
    # 删除多余的文件夹, 并将123 文件夹重命名为zero
    p = Path(save_path)
    want_words += '123'
    for i in p.iterdir():
        if i.name not in want_words:
            shutil.rmtree(i)
    p = p / '123'
    p.rename(p.parent/'zero')


if __name__ == "__main__":
    root_path = "../../data/"
    test_txt = root_path + "testing_list.txt"
    val_txt = root_path + "validation_list.txt"
    train_txt = root_path + "training_list.txt"
    want_words = 'one,yes,no,up,down,left,right,on,off,stop,go,nihaoxr,xrxr'
    save_path = "../../test_data"
    flag = True  # 判断是否需要所有文件参与训练和测试


    # if not os.path.exists(root_path):
    #     raise Exception("No " + root_path + "!")

    assert os.path.exists(root_path), Exception("No " + root_path + "!")


    # update saved test files
    for root, dirs, files in os.walk(root_path):
        for path in dirs:
            if flag:
                # all selected
                # 把路径写入text文件
                save_file(os.path.join(root, path), val_txt, test_txt, save_path, want_words)
            else:
                # TC-Net data split to .txt
                # only saved files with wanted words
                if path in want_words:
                    # 把路径写入text文件
                    save_file(os.path.join(root, path), val_txt, test_txt, save_path, want_words)
    #
    # save_other_to_zero(save_path, want_words)
    # modif_dir(save_path, want_words)
    #
    # # 统计有多少个test 文件
    # p = Path(save_path)
    # test_file_nums = len(list(p.glob('*/*.wav')))
    #
    # print(f"共有{test_file_nums}个测试文件")
    print("数据分类完成....")

    # p = Path(save_path)
    # p_list = list(p.glob('*/*.wav'))
    # print(len(p_list))  # 4551



