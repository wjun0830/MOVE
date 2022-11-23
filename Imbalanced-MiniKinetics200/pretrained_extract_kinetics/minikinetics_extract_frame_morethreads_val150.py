from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing
import time
import os
n_thread = 100


def vid2jpg_val_mp(val_file):
    class_name, video_name, start, end, split, is_cc = val_file.split(',')
    class_path = '/project/data/kinetics-dataset/minikinetics200/{}/{}'.format(split, class_name)
    # try:
    #     os.mkdir(class_path)
    # except OSError:
    #     print(class_path)
    #     pass
    videofile_name = video_name + '_{:06d}_{:06d}.mp4'.format(int(start), int(end))
    file_name = os.path.join(val_video_path, videofile_name)

    if '.mp4' not in file_name:
        return

    dst_directory_path = os.path.join(class_path, videofile_name.replace('.mp4',''))

    video_file_path = file_name
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')): # 한장도 없으면
                # pass
                # subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                # print('remove {}'.format(dst_directory_path))
                # os.mkdir(dst_directory_path)
                os.chmod(dst_directory_path, 0o777)
                return
            else: # 한장이라도있으면스
                # print('*** convert has been done: {}'.format(dst_directory_path))
                if not os.path.exists(os.path.join(dst_directory_path, 'img_00150.jpg')): #  한장이상 150장 이하면 다시 추출
                    os.chmod(dst_directory_path, 0o777)
                    # pass
                else: # 150장보다많으면 그냥 패
                    return
        else:
            os.mkdir(dst_directory_path)
            os.chmod(dst_directory_path, 0o777)
    except:
        print(dst_directory_path)
        return

    cmd = 'ffmpeg -y -i \"{}\" -vf \"minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\"  -threads 40 -qscale:v 2 \"{}/img_%05d.jpg\"'.format(
        video_file_path,
        dst_directory_path)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

train_list = open('/project/data/kinetics-dataset/mini-kinetics-200/kinetics_train.csv')
val_list = open('/project/data/kinetics-dataset/mini-kinetics-200/kinetics_val.csv')

train_video_path = '/project/data/kinetics-dataset/k400/train'
val_video_path = '/project/data/kinetics-dataset/k400/val'
# label,youtube_id,time_start,time_end,split,is_cc
_ = val_list.readline()
vlines = []
with val_list as vfile:
    for line in vfile:
        vlines.append(line) #storing everything in memory!

with multiprocessing.Pool(60) as p:
    p.map(vid2jpg_val_mp, vlines)

# _ = train_list.readline()
# tlines = []
# with train_list as tfile:
#     for line in tfile:
#         tlines.append(line) #storing everything in memory!
#
# with multiprocessing.Pool(50) as p:
#     p.map(vid2jpg_train_mp, tlines)