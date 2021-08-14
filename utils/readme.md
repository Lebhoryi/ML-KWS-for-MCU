> - lebhoryi 写的文件脚本存放处，文件中有使用说明
> - mail:  lebhoryi@rt-thread.mail


./ <br>
├── data_aug.py       # 语音增强 <br>
├── data_clips.py     # 将数据随机分10%到验证和测试的txt中，txt在data目录下 <br>
├── get_wav.py        # 从网页爬取语音数据 <br>
├── get_mfcc.py # tf1 获取单个audio的mfcc值 <br>
├── get_output_from_network.py  # 获取网络结构中的中间输出结果<br>
├── get_variable_name_from_ckpt.py  # 从ckpt 文件中获取变量的名字<br>
├── librosa_mfcc.py  # 用librosa 实现mfcc 提取 <br>
├── model.py  # 重构网络结构, 获取中间层的输出结果用 <br>
├── readme.md  # readme <br>
├── rm_aug_silence.py  # 移除audio 的开头和结尾的静音区 <br>
├── test_model.py  # 用自己的数据集对模型进行测试, 获取acc/precision/recall 等 <br>
├── tf2_mfccs.py  # tf2 分步骤获取mfcc,包括加窗/FFT等 <br>
└── wav8_16k.sh       # 强制将语音数据改为 16k 采样率，16 bits， 1 channel <br>
