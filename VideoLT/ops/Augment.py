# -*- coding: utf-8 -*-
import numpy as np
import torch

def mixup(input, target, gamma):
    '''
    Original mixup
    '''
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    perm_input = input[indices]
    perm_target = target[indices]
    mixed_input = input * gamma + perm_input * (1 - gamma)
    mixed_target = target * gamma + perm_target * (1 - gamma)    
    return mixed_input, mixed_target

def FrameStack(input, target, clip_length, ap):
    '''
    Func:
        dynamically sample frames from two videos according to their APs.
        For video with higher AP, sample fewer frames.
        For video with lower AP, sample more frames.

    Params:
        clip_length: the final length of video.

    '''
    input_size = input.size()

    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    perm_input = input[indices]
    perm_target = target[indices]
    
    assert (ap.size(0) == target.size(1)), \
        (print ("AP size = {} != target size = {}".format(ap.size(0), target.size(1))))
    
    ap = ap.cuda()
    sample_ap = torch.mul(target, ap)

    mean_ap = torch.mean(sample_ap, -1)
    perm_mean_ap = mean_ap[indices]

    mixed_inputs = []
    mixed_targets = []
    for i in range(input.size(0)):
        
        if (mean_ap[i] + perm_mean_ap[i]) < 1e-5:
            beta = 0.5
        else:
            beta = (mean_ap[i] / (mean_ap[i] + perm_mean_ap[i]))

        F_j = int(beta * clip_length)
        F_i = clip_length - F_j

        F_j_index = np.linspace(0, input_size[1] - 1, F_j)
        F_i_index = np.linspace(0, input_size[1] - 1, F_i)

        mixed_sample = torch.cat((perm_input[i][F_j_index], input[i][F_i_index]), dim=0)

        mixed_target = target[i] * (1 - beta) + perm_target[i] * beta

        mixed_inputs.append(mixed_sample)
        mixed_targets.append(mixed_target)

    mixed_inputs = torch.stack(mixed_inputs).cuda()

    mixed_targets = torch.stack(mixed_targets).cuda()

    return mixed_inputs, mixed_targets

def dynamic_extrapolation(feat1, feat2):
    ''' feat1, feat2 : bsz, 2048'''
    lam = np.random.beta(2.0, 2.0) + 1
    if isinstance(lam, np.ndarray):
        lam = torch.from_numpy(lam).float().cuda()
    feat = lam * feat1 + (1 - lam) * feat2
    return feat.unsqueeze(1)

def calibrated_interpolation(feat1, feat2, target, smooth_factor):
    gamma = np.random.beta(1.0, 1.0)
    indices = torch.randperm(feat2.size(0), device=feat2.device, dtype=torch.long)
    feat2 = feat2[indices]
    perm_target = target[indices]
    mixed_input = feat1 * gamma + feat2 * (1 - gamma)
    mixed_target = target * gamma + perm_target * (1 - gamma)

    calib_target = mixed_target * smooth_factor.expand(target.size(0), target.size(1)).cuda()
    calib_target = torch.clamp(calib_target, 0, 1)

    return mixed_input, calib_target