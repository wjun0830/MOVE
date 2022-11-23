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

train_class_list = glob('/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train/*')
val_class_list = glob('/project/data/kinetics-dataset/minikinetics_feature/ResNet50/val/*')
# print(train_class_list)
trainlist = open('./train_imbkinetics200_list.txt', 'w')
vallist = open('./val_imbkinetics200_list.txt', 'w')
for ti, tc in enumerate(train_class_list):
    _, p, d, k, m, R, v, featname = tc.split('/')
    featname = featname.replace('.npy','')
    trainlist.write(featname+'\n')
trainlist.flush()
trainlist.close()


for ti, tc in enumerate(val_class_list):
    _, p, d, k, m, R, v, featname = tc.split('/')
    featname = featname.replace('.npy','')
    vallist.write(featname+'\n')
vallist.flush()
vallist.close()