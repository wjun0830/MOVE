import os
import copy
import torch
import shutil
import time
import warnings
import sys
import numpy as np
import random
from ops import Augment
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from opts import parser
from ops.mapmeter import mAPMeter, LTMeter
from ops.utils import AverageMeter, accuracy, LTAverageMeter, perclsaccuracy, LTconfMeter

from ops import losses
from tools import utils

from dataset import dutils
from models import models
from ops.feature_loader import BasicDataset, ResamplingDataset_Mask
import torch.nn as nn
from models.net_module import NetVLAGD

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay

    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))

    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
import math
def load_data(num_class, input_dir):
    train_list = open(args.train_list, 'r').readlines()
    val_list = open(args.val_list, 'r').readlines()
    if args.resample == 'None':
        train_dataset = BasicDataset(train_list, input_dir, args.train_num_frames,\
                     cls_num=num_class, train_mode=True, multilabel=True)
    else:
        train_dataset = ResamplingDataset_Mask(train_list, input_dir, args.train_num_frames, \
                     rstype=args.resample, cls_num=args.num_class, train_mode=True)
    cls_num_list = train_dataset.cls_num_list
    val_dataset = BasicDataset(val_list, input_dir, args.val_num_frames, \
                     cls_num=num_class, train_mode=False, multilabel=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, \
                        shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    cls_num_list = np.array([num / sum(cls_num_list) for num in cls_num_list])
    cls_num_list = (cls_num_list - cls_num_list.min()) / (cls_num_list.max() - cls_num_list.min())

    sampling_prob = cls_num_list * args.train_num_frames
    sampling_prob[sampling_prob < args.lb] = args.lb

    target_smooth_factor = 1 - cls_num_list
    target_smooth_factor += args.calib_bias
    smooth_factor = torch.FloatTensor(target_smooth_factor)
    return train_dataloader, val_dataloader, smooth_factor

def main():
   
    global args, best_mAP, criterion, optimizer, tf_writer, log_training

    
    best_mAP = 0

    args = parser.parse_args()
    start_epoch = args.start_epoch     
    num_class = args.num_class
    if args.resample != 'None':
        args.reduce = "none"
    print ("########################################################################\n")
    print ("Feature name: {} \nNumber of class: {} \nTrain frames: {} \nVal frames: {}\nReduction: {}".\
            format(args.feature_name, args.num_class, args.train_num_frames, args.val_num_frames, args.reduce))
    print ("Applied long-tailed strategies: \n")
    print ("\tAugmentation: {} \t Re-weighting: {} \t Re-sampling: {} \n". \
            format(args.augment, args.loss_func, args.resample))
    print ("######################################################################## \n") 
    check_rootfolders()
    setup_seed(args.seed)

    input_dir =  dutils.get_feature_path(args.feature_name)
    feature_dim = dutils.get_feature_dim(args.feature_name)
    args.lc_list, args.train_list, args.val_list = dutils.get_label_path()

    train_loader, val_loader, smooth_factor = load_data(num_class, input_dir)

    criterion = utils.find_class_by_name(args.loss_func, [losses])(args, logits=True, reduce=args.reduce)
    
    indices = utils.get_indices(args.lc_list, head=args.head, tail=args.tail)
    
    model = utils.find_class_by_name(args.model_name, [models])(3072, num_class)
    model = model.cuda()

    vladmodel = utils.find_class_by_name('NetVLADModel_Aggregator', [models])(feature_dim, num_class)
    vladmodel = vladmodel.cuda()
    MHA = utils.find_class_by_name('MultiHeadAttention_Aggregator', [models])(n_head=4, d_model=2048, d_k=512, d_v=512, dropout=0.1).cuda()
    if args.resume != "": 
        print ("=> Loading checkpoint {}".format(args.resume))
        
        ckpt = torch.load(args.resume)
        best_mAP = ckpt['best_mAP']
        start_epoch = ckpt['epoch'] + 1
        acc1 = ckpt['Acc@1']
        acc5 = ckpt['Acc@5']
        sd = ckpt['state_dict']
        sdMHA = ckpt['MHA']
        sdvlad = ckpt['vlad']
        print ("Loaded checkpoint {} epoch {}: best_mAP {} | Acc@1 {} | Acc@5 {}". \
                format(args.resume, start_epoch, best_mAP, acc1, acc5))
    
        model.load_state_dict(sd)
        MHA.load_state_dict(sdMHA)
        vladmodel.load_state_dict(sdvlad)
    if args.evaluate:
        if args.resume != "": #
            tf_writer=None
            acc1, acc5, mAP = validate(val_loader, model, vladmodel, 0, None, indices, MHA)



        else:
            print('please enter args.resume with args.evaluate')

        exit(0)

    print ("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print ('\t', name)
    for name, param in MHA.named_parameters():
        params_to_update.append(param)
        print ('\t', name)

    for name, param in vladmodel.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
    optimizer = torch.optim.Adam(params_to_update, lr=args.lr)
    
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'),'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        print ("Training for Epoch {}".format(epoch))
        if args.resample != "None":
            rs_train(train_loader, model, vladmodel, epoch, log_training, MHA)
        else:
            train(train_loader, model, vladmodel, epoch, log_training, MHA, smooth_factor)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            acc1, acc5, mAP = validate(val_loader, model, vladmodel, epoch, log_training, indices, MHA)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            tf_writer.add_scalar('best_mAP/test_best', best_mAP, epoch)
            
            print ('Test Epoch {}: Acc@1: {} | Acc@5: {} | mAP: {} | best_mAP: {}'.\
                    format(epoch, acc1, acc5, mAP, best_mAP))

            save_checkpoint({
            'epoch': epoch + 1,
            'feature': args.feature_name,
            'state_dict': model.state_dict(),
            'MHA': MHA.state_dict(),
            'vlad': vladmodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP,
            'Acc@1': acc1,
            'Acc@5': acc5},
            is_best)

def train(loader, model, vladmodel, epoch, log, MHA, smooth_factor):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()

    model.train()
    vladmodel.train()
    end = time.time()
    MHA.train()

    if args.loss_func == 'LDAM':
        # apply DRW to LDAM
        criterion.reset_epoch(epoch)
    for i, (vid, feature, target, index, label) in enumerate(loader):
        real_labels = [np.array(i.split(',')).astype(np.uint32) for i in label]
        feature = feature.cuda()
        target = target.float().cuda(non_blocking=True)
        v_feature = vladmodel(feature).unsqueeze(1)
        q_feature, attn = MHA(feature, feature, feature)
        q_feature = torch.cat([v_feature, q_feature], dim=2)


        if args.augment == "mixup":
            gamma = np.random.beta(1.0, 1.0)
            mixed_input, mixed_target = Augment.mixup(q_feature, target, gamma)
            prediction, output = model(mixed_input)
            loss = criterion(output, mixed_target)
        elif args.augment == "None":
            prediction, output = model(q_feature)
            loss = criterion(output, target)
        else:
            print ("{} not implemented. Please choose ['mixup', 'None'].".\
                    format(args.augment))
            raise NotImplementedError


        losses.update(loss.item(), output.size(0))

        with torch.no_grad():
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1, output.size(0))
            top5.update(prec5, output.size(0))
        
        # accumulate gradient for each parameter
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        # update parameters based on current gradients
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                      .format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, \
                lr=optimizer.param_groups[-1]['lr']))

            sys.stdout.write('\r')
            sys.stdout.write(output)
            sys.stdout.flush()
            # print(output)
            
            log.write(output)
            log.flush()
    
    tf_writer.add_scalar('loss/train_epoch', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(loader, model, vladmodel, epoch, log, indices, MHA):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()
    LTtop1 = LTAverageMeter(indices)
    LTtop5 = LTAverageMeter(indices)
    LTmAP =LTMeter(indices)


    LTconf = LTconfMeter(indices)
    model.eval()
    MHA.eval()
    vladmodel.eval()

    end = time.time()
    with torch.no_grad():
        for i, (vid, feature, target, index, label) in enumerate(loader):
            label = [np.array(i.split(',')).astype(np.uint32) for i in label]
            feature = feature.cuda()

            target = target.float().cuda()

            target = target.float().cuda(non_blocking=True)
            v_feature = vladmodel(feature).unsqueeze(1)
            q_feature, attn = MHA(feature, feature, feature)
            q_feature = torch.cat([v_feature, q_feature], dim=2)
            prediction, output = model(q_feature)
            
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            corr1, corr5 = perclsaccuracy(output.data, target, topk=(1, 5))

            if args.resample != "None":
                loss = torch.mean(torch.sum(loss, 1))
            losses.update(loss.item(), feature.size(0))
            LTconf.update(prediction, label)
            top1.update(prec1, feature.size(0))
            top5.update(prec5, feature.size(0))
            LTtop1.update(corr1, label)
            LTtop5.update(corr5, label)
            mAP.add(prediction, target)
            LTmAP.add(prediction, target)
           
     
            batch_time.update(time.time() - end)
            end = time.time()

    head_map = LTmAP.value()["head"]
    medium_map = LTmAP.value()["medium"]
    tail_map = LTmAP.value()["tail"]


    output = ('Testing Results: Accuracy Prec@1,5 {top1.avg:.5f} {top5.avg:.5f} | Loss {loss.avg:.5f} '
              .format(top1=top1, top5=top5, loss=losses))

    lt_accoutput = (
        'H@1 {htop1:.5f} H@5 {htop5:.5f} | M@1 {mtop1:.5f} M@5 {mtop5:.5f} | T@1 {ttop1:.5f} T@5 {ttop5:.5f}'
            .format(htop1=LTtop1.value()["head"], htop5=LTtop5.value()["head"], mtop1=LTtop1.value()["medium"],
                    mtop5=LTtop5.value()["medium"], ttop1=LTtop1.value()["tail"], ttop5=LTtop5.value()["tail"]))


    print('\n',lt_accoutput)
    print(output)
    lt_output = (
        "Overall, Head, Medium, Tail mAP, Acc@1, 5 = {:.3f} {:.5f} {:.5f} {:.5f} {top1.avg:.5f} {top5.avg:.5f}". \
        format(mAP.avg(), head_map, medium_map, tail_map, top1=top1, top5=top5))
    print (lt_output)


    head_conf = LTconf.value()["head"]
    medium_conf = LTconf.value()["medium"]
    tail_conf = LTconf.value()["tail"]
    head_cconf = LTconf.value()["head_c"]
    medium_cconf = LTconf.value()["medium_c"]
    tail_cconf = LTconf.value()["tail_c"]

    lt_confoutput = (
        'Confidence : Head {head_conf:.5f} | Medium {medium_conf:.5f} | Tail {tail_conf:.5f} \n Correct Conf : cHead {head_cconf:.5f} | Medium {medium_cconf:.5f} | Tail {tail_cconf:.5f}'
            .format(head_conf=head_conf, medium_conf=medium_conf, tail_conf=tail_conf,
                    head_cconf=head_cconf, medium_cconf=medium_cconf, tail_cconf=tail_cconf))
    print(lt_confoutput)
    if log is not None:
        log.write(output + '  mAP {}\n'.format(mAP.avg()))
        log.write(lt_output+'\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
        tf_writer.add_scalar('mAP/test', mAP.avg(), epoch)
    return top1.avg, top5.avg, mAP.avg()


def rs_train(loader, model, vladmodel, epoch, log, MHA):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()
    MHA.train()
    model.train() 
    end = time.time()
    vladmodel.train()
    if args.loss_func == 'LDAM':
        # apply DRW to LDAM
        criterion.reset_epoch(epoch)
    for i, (vid, feature, target, mask) in enumerate(loader):
        feature = feature.cuda()
        target = target.float().cuda(non_blocking=True)
        v_feature = vladmodel(feature).unsqueeze(1)
        q_feature, attn = MHA(feature, feature, feature)
        q_feature = torch.cat([v_feature, q_feature], dim=2)

        if args.augment == "mixup":
            gamma = np.random.beta(1.0, 1.0)
            mixed_input, mixed_target = Augment.mixup(q_feature, target, gamma)
            prediction, output = model(mixed_input)
            loss = criterion(output, mixed_target)
        elif args.augment == "None":
            prediction, output = model(q_feature)
            loss = criterion(output, target)
        else:
            print("{} not implemented. Please choose ['mixup', 'None'].". \
                  format(args.augment))
            raise NotImplementedError


        mask = mask.float().cuda()

        loss = loss * mask
        loss = torch.mean(torch.sum(loss, 1))
        losses.update(loss.item(), output.size(0))

        with torch.no_grad():
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1, output.size(0))
            top5.update(prec5, output.size(0))
        
        # accumulate gradient for each parameter
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        # update parameters based on current gradients
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                      .format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, \
                lr=optimizer.param_groups[-1]['lr']))
            sys.stdout.write('\r')
            sys.stdout.write(output)
            sys.stdout.flush()
            # print(output)

            
            log.write(output)
            log.flush()
    
    tf_writer.add_scalar('loss/train_epoch', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


if __name__=='__main__':
    main()
