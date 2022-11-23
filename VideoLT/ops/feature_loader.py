import os
import numpy as np
import math
import random
import torch
from numpy.random import randint
from torch.utils.data import Dataset
import copy
class BasicDataset(Dataset):
    def __init__(self, data_list, input_dir, num_frames, cls_num=1004, train_mode=True, multilabel=False, onlyhead=False, headmax=-1, lb=3.0, motionlb=3.0):
        self.data_list = data_list
        self.input_dir = input_dir
        self.num_frames = num_frames
        self.cls_num = cls_num
        self.table = np.eye(cls_num)
        self.train_mode = train_mode
        self.multilabel = multilabel
        self.onlyhead = onlyhead
        self.cls2vid = {}
        self.vid2cls = {}

        self.num_list = np.zeros(cls_num)
        for data in self.data_list:
            vid_id, label = data.split()
            for label in [int(i) for i in label.split(',')]:
                if label not in self.cls2vid:
                    self.cls2vid[label] = []
                if vid_id not in self.vid2cls:
                    self.vid2cls[vid_id] = []
                self.num_list[label] += 1
                self.cls2vid[label].append(vid_id)
                self.vid2cls[vid_id].append(label)
        if self.onlyhead:
            headidx = np.where(self.num_list > 500)[0]
        self.pos_weight = self.num_list
        self.numberofsamples = copy.deepcopy(self.num_list)

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, self.pos_weight)
        weight_for_pos = (1.0 - beta) / np.array(effective_num)
        weight_for_pos = weight_for_pos / np.sum(weight_for_pos) * len(self.pos_weight)
        self.weight_for_pos = torch.FloatTensor(weight_for_pos).cuda()
        self.pos_weight = self.pos_weight / self.pos_weight.sum()
        self.pos_weight = torch.FloatTensor(self.pos_weight).cuda()

        self.cls_num_list = copy.deepcopy(self.num_list)
        self.num_list = [math.sqrt(num) for num in self.num_list]


    def __getitem__(self, index):
        data = self.data_list[index]
        vid_id, label = data.split()

        feature = np.load(os.path.join(self.input_dir, '%s.npy'%(vid_id)))
        if self.train_mode:
            feature, index = self.random_sample(feature, self.num_frames)
        else:
            feature, index = self.uniform_sample(feature, self.num_frames)

        onehotlabel = np.sum(self.table[
            [int(i) for i in label.split(',')]
        ], axis=0)

        if self.multilabel:
            return vid_id, feature, onehotlabel, index, label
        return vid_id, feature, onehotlabel, index
    
    def __len__(self):
        # print('total data loader', len(self.data_list))
        return len(self.data_list)

    def random_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.sort(randint(frames, size=uniform_length))
        feature = feature[index]
        return feature, index

    def uniform_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.linspace(0, frames - 1, uniform_length).astype(np.int32)
        feature = feature[index]
        return feature, index





class ResamplingDataset_Mask(Dataset):
    def __init__(self, data_list, input_dir, num_frames, rstype, cls_num=1004, train_mode=True, multilabel=False):
        self.data_list = data_list
        self.input_dir = input_dir
        self.num_frames = num_frames
        self.cls_num = cls_num
        self.table = np.eye(cls_num)
        self.train_mode = train_mode
        self.rstype = rstype
        self.multilabel = multilabel
        # construct mapping for { cls -> vid }
        self.cls2vid = {}
        self.vid2cls = {}

        self.num_list = np.zeros(cls_num)
        for data in self.data_list:
            vid_id, label = data.split()
            for label in [int(i) for i in label.split(',')]:
                if label not in self.cls2vid:
                    self.cls2vid[label] = []
                if vid_id not in self.vid2cls:
                    self.vid2cls[vid_id] = []
                self.num_list[label] += 1
                self.cls2vid[label].append(vid_id)
                self.vid2cls[vid_id].append(label)

        self.cls_num_list = copy.deepcopy(self.num_list)
        self.num_list = [math.sqrt(num) for num in self.num_list]
        self.square_p = np.array([num/sum(self.num_list) for num in self.num_list])
        self.square_p = (self.square_p - self.square_p.min()) / (self.square_p.max() - self.square_p.min())
        self.square_p = self.square_p * 60
        self.square_p[self.square_p < 1.0] = 1.0


    def __getitem__(self, index):

        if self.rstype == "CBS":
            select_label = random.randint(0, self.cls_num-1)
        elif self.rstype == "SRS":
            select_label = np.random.choice(np.arange(self.cls_num), p=self.square_p)
        else:
            print ("{} not implemented. Please select ['CBS', 'SRS'].")
            raise NotImplementedError

        vidlist = (self.cls2vid[select_label])
        select_vid_idx = random.randint(0, len(vidlist) - 1)
        vid_id = vidlist[select_vid_idx]
        cls = self.vid2cls[vid_id]
        
        feature = np.load(os.path.join(self.input_dir, '%s.npy'%(vid_id)))

        if self.train_mode:
            feature = self.random_sample(feature, self.num_frames)
        else:
            feature = self.uniform_sample(feature, self.num_frames)

        label = np.sum(self.table[cls], axis=0)
        mask = 1.0 - (label - self.table[select_label])
        # print(cls)
        if self.multilabel:
            return vid_id, feature, label, mask, cls

        return vid_id, feature, label, mask
    
    def __len__(self):
        return len(self.data_list)

    def random_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.sort(randint(frames, size=uniform_length))
        feature = feature[index]
        return feature

    def uniform_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.linspace(0, frames - 1, uniform_length).astype(np.int32)
        feature = feature[index]
        return feature






