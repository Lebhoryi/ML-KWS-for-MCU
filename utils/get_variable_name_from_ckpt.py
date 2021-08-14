# coding=utf-8
'''
@ Summary: 查看ckpb 文件中的变量
@ Update:  

@ file:    123.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/5/21 下午5:52
'''

import os
from tensorflow.python import pywrap_tensorflow

# current_path = os.getcwd()
# model_dir = os.path.join(current_path, 'train_model')
# checkpoint_path = os.path.join(model_dir,'521_cnn/best/cnn_8578.ckpt-9600')

checkpoint_path = '/home/lebhoryi/RT-Thread/WakeUp-Xiaorui/tensorflow_train-master/' \
                  'train_model/525_dscnn/best/ds_cnn_9224.ckpt-29600_bnfused'

# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

tensor_name_list = [key for key in var_to_shape_map if 'Adam' not in key]

# Print tensor name and values
for key in var_to_shape_map:
    print(key)

# tensor_name_list = sorted(tensor_name_list)
# for i in tensor_name_list:
#     print(i)
