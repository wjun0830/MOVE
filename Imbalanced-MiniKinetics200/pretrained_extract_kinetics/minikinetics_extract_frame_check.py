from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing
import time
import os
from glob import glob

train_class_list = glob('/project/data/kinetics-dataset/minikinetics200/train/*')
val_class_list = glob('/project/data/kinetics-dataset/minikinetics200/val/*')
train_flag = 0
val_flag = 0
train_num_list = []
small_train_num_list = []
val_num_list = []
small_val_num_list = []
for ti, tc in enumerate(train_class_list):
    cnt = 0
    small_cnt = 0
    tcvl = glob(tc + '/*')
    for tcv in tcvl:
        tcvfl = glob(tcv + '/*')
        if len(tcvfl) > 10:
            cnt += 1
        if len(tcvfl) > 1 and len(tcvfl) <= 10:
            small_cnt += 1
    print('train ['+str(ti)+ '/' + str(len(train_class_list)) + ']' + str(cnt) + ' / ' + str(len(tcvl)) + ' done **' + str(small_cnt) + 'less than 10 frames')
    train_num_list.append(cnt)
    small_train_num_list.append(small_cnt)
    if cnt == len(tcvl):
        pass
    else:
        train_flag += 1

for vi, vc in enumerate(val_class_list):
    cnt = 0
    small_cnt = 0
    vcvl = glob(vc + '/*')
    for vcv in vcvl:
        vcvfl = glob(vcv + '/*')
        if len(vcvfl) > 10:
            cnt += 1
    print('val ['+str(vi)+ '/' + str(len(val_class_list)) + ']' + str(cnt) + ' / ' + str(len(vcvl)) + ' done **' + str(small_cnt) + 'less than 10 frames')
    val_num_list.append(cnt)
    small_val_num_list.append(small_cnt)
    if cnt == len(vcvl):
        pass
    else:
        val_flag += 1
import numpy as np
small_train_num_list = np.array(small_train_num_list)
small_val_num_list = np.array(small_val_num_list)
print('train less than 10 frames exist')
print(small_train_num_list.any())
print('val less than 10 frames exist')
print(small_val_num_list.any())

if train_flag == 0:
    print('train set Frame conversion done !!! OOOOO')
else:
    print('train set Frame Conversion Not done !!! XXXXX')

if val_flag == 0:
    print('val set Frame conversion done !!! OOOOO')
else:
    print('val set Frame Conversion Not done !!! XXXXX')

import numpy as np
import matplotlib.pyplot as plt
train_num_list = np.array(train_num_list)
y = np.arange(200).astype(np.int32) + 1
plt.plot(y, train_num_list, label='linear', color='navy')

plt.xticks([], [])
plt.yticks([], [])
# ax.set_xticks([])
# ax.set_xticks([], minor=True)
plt.legend(loc='upper right')
# plt.axis('off')
plt.savefig('/project/minikinetics_train_num.png')
plt.close()
# train_list = open('/project/data/kinetics-dataset/mini-kinetics-200/kinetics_train.csv')
# val_list = open('/project/data/kinetics-dataset/mini-kinetics-200/kinetics_val.csv')
#
# train_video_path = '/project/data/kinetics-dataset/k400/train'
# val_video_path = '/project/data/kinetics-dataset/k400/val'
# # label,youtube_id,time_start,time_end,split,is_cc
# _ = val_list.readline()
# vlines = []
# with val_list as vfile:
#     for line in vfile:
#         vlines.append(line) #storing everything in memory!
#
# # with multiprocessing.Pool(60) as p:
# #     p.map(vid2jpg_val_mp, vlines)
#
# _ = train_list.readline()
# tlines = []
# with train_list as tfile:
#     for line in tfile:
#         tlines.append(line) #storing everything in memory!
#
# # with multiprocessing.Pool(50) as p:
# #     p.map(vid2jpg_train_mp, tlines)