# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import utils
import numpy as np
from mini_loader import *
import glob
import warnings
import math
import sys
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=101, type=int,
                    help='depth of the network (default: 50)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100



def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    args.MODEL_NAME = 'ResNet' + str(args.depth)
    val_data_list = glob.glob('/project/data/kinetics-dataset/minikinetics200/val/*/*/')
    train_data_list = glob.glob('/project/data/kinetics-dataset/minikinetics200/train/*/*/')
    # print(train_data_list)
    def get_query_transforms():

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_dataset = MiniKineticsDataset(val_data_list, transform=get_query_transforms(), num_frames=150, cls_num=200,
                                      mode='val', depth=args.depth)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, \
                                                 shuffle=False, pin_memory=True)
    train_dataset = MiniKineticsDataset(train_data_list, transform=get_query_transforms(), num_frames=150, cls_num=200,
                                      mode='train', depth=args.depth)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, \
                                                 shuffle=False, pin_memory=True)


    numberofclass = 1000


    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
        if args.depth == 50:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-19c8e357.pth', map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict, strict=False)
        elif args.depth == 101:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict, strict=False)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    featsave(train_loader, model, mode='train', args=args)
    featsave(val_loader, model, mode='val', args=args)

def featsave(loader, model, mode='train', args=None):
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, feature_name) in enumerate(loader):
        input = input.cuda() # torch.Size([1, 150, 3, 224, 224])
        input = input.reshape(150, 3, 224, 224)
        output = model(input, feat=True) # 150 2048
        # print(feature_name)
        np.save('/project/data/kinetics-dataset/minikinetics_feature/' + args.MODEL_NAME + '/' + mode + '/' + feature_name[0], output.clone().detach().cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)


        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
