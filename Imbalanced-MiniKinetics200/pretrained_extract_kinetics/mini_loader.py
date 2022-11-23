import os
import numpy as np
import math
import random
import torch
from numpy.random import randint
from torch.utils.data import Dataset
import copy
from torch.utils.data import Dataset
from PIL import Image
import json
import glob
from torchvision.transforms import transforms
import pickle
def find_classes(directory):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx # List[str], Dict[str, int]

class MiniKineticsDataset(Dataset):
    def __init__(self, data_list, transform, num_frames=150, cls_num=200, mode='train', depth=50):
        self.data_list = data_list
        self.num_frames = num_frames
        self.cls_num = cls_num
        self.train_mode = mode
        self.transform = transform
        self.depth=depth
        self.cls2vid = {}
        self.vid2cls = {}
        list_txt = open('/project/data/kinetics-dataset/minikinetics200/' + mode + '.lst', 'w')
        os.chmod('/project/data/kinetics-dataset/minikinetics200/' + mode + '.lst', 0o777)
        classes, class_to_idx = find_classes('/project/data/kinetics-dataset/minikinetics200/{}'.format(self.train_mode))
        self.classes = classes
        self.class_to_idx = class_to_idx
        with open('/project/data/kinetics-dataset/minikinetics200/' + mode + '_class_to_idx.pkl', 'wb') as f:
            pickle.dump(class_to_idx, f)
        # print(class_to_idx) 'folding paper': 60, 'front raises': 61, 'giving or receiving award': 62, 'golf driving': 63, 'golf putting': 64}
        # data_list : '/project/data/kinetics-dataset/minikinetics200/val/golf putting/rHP-OmX2hBo_000020_000030/'
        for data in self.data_list:
            _, _, _, _, _, split, class_name, feature_name,_ = data.split('/')
            list_txt.write("%s %d\n" % (feature_name, class_to_idx[class_name]))
            # print(split, class_name, feature_name) val abseiling YgzCJmncmUA_000027_000037
        list_txt.flush()
        list_txt.close()

    def __getitem__(self, index):
        data = self.data_list[index] # data : '/project/data/kinetics-dataset/minikinetics200/val/golf putting/rHP-OmX2hBo_000020_000030/'
        _, _, _, _, _, split, class_name, feature_name, _ = data.split('/')
        # print(data)
        # print(data)
        framelist = np.array(glob.glob(data+'*'))
        if len(framelist) > 150:
            frame_list, index = self.uniform_sample(framelist, self.num_frames)
            delete_idx = list(set(list(range(0, len(framelist)))) - set(index))
            for deleteframe in framelist[delete_idx]:
                os.remove(deleteframe)
                # print(deleteframe, i)
            # print(len(framelist), len(index), len(delete_idx))
            # print(framelist[delete_idx][0])
            # exit(1)
        elif len(framelist) == 150:
            frame_list, index = self.uniform_sample(framelist, self.num_frames)
        elif len(framelist) > 10:
            frame_list, index = self.uniform_repeat(framelist, self.num_frames)
        else:
            if self.train_mode == 'train':
                print('train empty frame : ' + data)
            if self.train_mode != 'train':
                print('val empty frame : ' + data)
            return

        videoframes = []
        for fr in frame_list:
            img = Image.open(fr).convert("RGB")
            img = self.transform(img)
            videoframes.append(img)
            # print(img.shape)
            # print(fr) /project/data/kinetics-dataset/minikinetics200/val/driving tractor/FC9yd-arj7A_000077_000087/img_00001.jpg
            # exit(1)
        videoframes = torch.stack(videoframes, dim=0)
        # print(videoframes.shape) # 150, 3, 224, 224
        # exit(1)
        return videoframes, feature_name
    
    def __len__(self):
        # print('total data loader', len(self.data_list))
        return len(self.data_list)

    def random_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.sort(randint(frames, size=uniform_length))
        feature = feature[index]
        return feature, index

    def uniform_sample(self, frame_list, uniform_length):
        frames = len(frame_list)
        index = np.linspace(0, frames - 1, uniform_length).astype(np.int32)
        frame_list = frame_list[index]
        return frame_list, index

    def uniform_repeat(self, frame_list, uniform_length):
        frames = len(frame_list)
        index = np.linspace(0, frames - 1, uniform_length).astype(np.int32)
        frame_list = frame_list[index]
        return frame_list, index



if __name__ == '__main__':
    import glob
    val_data_list = glob.glob('/project/data/kinetics-dataset/minikinetics200/val/*/*/')
    train_data_list=glob.glob('/project/data/kinetics-dataset/minikinetics200/train/*/*/')


    def get_query_transforms():

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_dataset = MiniKineticsDataset(val_data_list, transform=get_query_transforms(), num_frames=150, cls_num=200, mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, \
                        shuffle=True, num_workers=12, pin_memory=True)

    for i, (data, class_name) in enumerate(val_dataloader):
        print(data.shape) #torch.Size([1, 150, 3, 224, 224])
        exit(1)
        continue


