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

for ti, tc in enumerate(train_class_list):
    tcvl = glob(tc + '/*') # search train class video list name
    for tcv in tcvl:
        tcvfl = glob(tcv + '/*') # search train class video frames
        if len(tcvfl) <= 10:
            os.rmdir(tcv)

#
for vi, vc in enumerate(val_class_list):
    vcvl = glob(vc + '/*')
    for vcv in vcvl:
        vcvfl = glob(vcv + '/*')
        if len(vcvfl) <= 10:
            os.rmdir(vcv)
