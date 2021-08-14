#!/bin/bash
# using：
# 将这个文件copy到对应的wav文件夹下面,用完记得删除
for file in *.wav; do
    #echo $file
    c=${file}
    #echo $c
    sox $c -c 1 -b 16 -r 16000 new_$c
    rm -f $c
    mv new_$c $c
done
