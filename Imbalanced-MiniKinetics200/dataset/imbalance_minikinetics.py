import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os
from numpy.random import randint
class imbalance_minikinetics(Dataset):
    cls_num = 200
    samples = 400
    def __init__(self, root, input_dir='/project/data/kinetics-dataset/minikinetics_feature/train',imb_type='exp', imb_factor=0.1, num_frames=60, rand_number=0,
                 transform=None, target_transform=None, cls_num=200):
        # super(imbalance_minikinetics, self).__init__(root, transform, target_transform)
        np.random.seed(rand_number)
        self.root = root
        self.input_dir = input_dir
        self.samples = []
        self.table = np.eye(cls_num)
        self.targets = []
        self.num_frames = num_frames #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        num_samples = np.zeros(200)
        for data in open(self.root):
            feature_name, class_label = data.split(' ')
            num_samples[int(class_label)] += 1
        self.img_max = num_samples.max()
        HeadtoTail_idx = num_samples.argsort()[::-1]

        self.Class_converter = dict()
        for idx, i in enumerate(HeadtoTail_idx):
            self.Class_converter[i] = idx

        for data in open(self.root):
            feature_name, class_label = data.split(' ')
            self.samples.append(feature_name)
            self.targets.append(self.Class_converter[int(class_label)])

        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)

        self.gen_imbalanced_data(self.img_num_list)
        self.num_list = np.zeros(cls_num)
        for cls_label in self.targets:
            self.num_list[cls_label] += 1

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = self.img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(self.img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            ### If specific class does not have enough samples,
            if len(idx) < the_img_num:
                the_img_num = len(idx)
            self.num_per_cls_dict[the_class] = the_img_num
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            #print(self.samples)
            res_list = [self.samples[i] for i in selec_idx]
            #print(res_list)
            new_data.extend(res_list)
            new_targets.extend([the_class, ] * the_img_num)
        #new_data = np.vstack(new_data)
        self.samples = new_data
        self.targets = new_targets

        
    def __len__(self):
        # print('total data loader', len(self.data_list))
        return len(self.samples)

    def __getitem__(self, index):
        data_path, label = self.samples[index], self.targets[index]

        feature = np.load(os.path.join(self.input_dir, '%s.npy' % (data_path)))
        feature, index = self.random_sample(feature, self.num_frames)
        # else:
        #     feature, index = self.uniform_sample(feature, self.num_frames)
        # onehotlabel = np.sum(self.table[
        #                          [int(i) for i in label.split(',')]
        #                      ], axis=0)
        onehotlabel = np.sum(self.table[
                                 [int(label)]
                             ], axis=0) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # if self.multilabel:
        return data_path, feature, onehotlabel, index, label
        # return vid_id, feature, onehotlabel, index

    def random_sample(self, feature, uniform_length):
        frames = feature.shape[0]
        index = np.sort(randint(frames, size=uniform_length))
        feature = feature[index]
        return feature, index


class TestDataset(Dataset):
    def __init__(self, root, input_dir='/project/data/kinetics-dataset/minikinetics_feature/val', cls_num=200, Class_converter=None):
        self.root = root
        self.input_dir = input_dir
        self.cls_num = cls_num
        self.table = np.eye(cls_num)
        self.Class_converter = Class_converter
        self.cls2vid = {}
        self.vid2cls = {}
        self.samples = []

        self.targets = []
        for data in open(self.root):
            feature_name, class_label = data.split(' ')
            self.samples.append(feature_name)
            self.targets.append(self.Class_converter[int(class_label)])


    def __getitem__(self, index):
        data_path, label = self.samples[index], self.targets[index]

        feature = np.load(os.path.join(self.input_dir, '%s.npy' % (data_path)))
        feature, index = self.uniform_sample(feature, 150)

        # onehotlabel = np.sum(self.table[
        #                          [int(i) for i in label.split(',')]
        #                      ], axis=0)
        onehotlabel = np.sum(self.table[
                                 [int(label)]
                             ], axis=0) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # if self.multilabel:
        return data_path, feature, onehotlabel, index, label

    def __len__(self):
        # print('total data loader', len(self.data_list))
        return len(self.targets)

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

def find_classes(directory):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx # List[str], Dict[str, int]

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = imbalance_minikinetics(root='/project/data/kinetics-dataset/minikinetics200/train.lst',
                                           input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train')
    val_dataset = TestDataset(root='/project/data/kinetics-dataset/minikinetics200/val.lst',
                                      input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/val',
                         Class_converter = train_dataset.Class_converter)

    classes, class_to_idx = find_classes('/project/data/kinetics-dataset/minikinetics200/train')
    idx_to_imbidx = train_dataset.Class_converter

    inv_idx_to_imbidx = {v: k for k, v in idx_to_imbidx.items()}
    inv_class_to_idx = {v: k for k, v in class_to_idx.items()}

    imbidx_to_classname = dict()
    for i in range(200):
        imbidx_to_classname[i] = inv_class_to_idx[inv_idx_to_imbidx[i]]
    # print('original class index')
    # print(inv_class_to_idx)
    # print('sorted class index')
    # print(imbidx_to_classname)


    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3, \
                                             shuffle=False, pin_memory=True)
    # for data_path, videoframes, onehotlabel, index, label in val_loader:
    #     # print(data_path)
    #     # print(videoframes.shape)
    #     # print(onehotlabel)
    #     # print(index)
    #     # print(label)
    #     continue
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, \
    #                                            shuffle=False, pin_memory=True)
    # for data_path, videoframes, onehotlabel, index, label in train_loader:
    #     # print(data_path)
    #     # print(videoframes.shape)
    #     # print(onehotlabel)
    #     # print(index)
    #     # print(label)
    #     continue

    td001 = imbalance_minikinetics(root='/project/data/kinetics-dataset/minikinetics200/train.lst',
                                           input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train',
                                           imb_factor=0.01)
    td002 = imbalance_minikinetics(root='/project/data/kinetics-dataset/minikinetics200/train.lst',
                                   input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train',
                                   imb_factor=0.02)
    td005 = imbalance_minikinetics(root='/project/data/kinetics-dataset/minikinetics200/train.lst',
                                   input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train',
                                   imb_factor=0.05)
    td01 = imbalance_minikinetics(root='/project/data/kinetics-dataset/minikinetics200/train.lst',
                                   input_dir='/project/data/kinetics-dataset/minikinetics_feature/ResNet50/train',
                                   imb_factor=0.1)
    print('imb ratio 001')
    print(td001.img_num_list)
    print('imb ratio 002')
    print(td002.img_num_list)
    print('imb ratio 005')
    print(td005.img_num_list)
    print('imb ratio 01')
    print(td01.img_num_list)