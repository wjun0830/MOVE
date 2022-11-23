import numpy as np
import torch

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


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


class LTAverageMeter(object):
    def __init__(self, indices):
        super(LTAverageMeter, self).__init__()
        self.tail, self.medium, self.head = indices
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

    def reset(self):
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

    def update(self, val, target, n=1):
        for idx, ml in enumerate(target):
            self.val[int(ml)] = val[idx]
            self.sum[int(ml)] += val[idx]
            self.count[int(ml)] += 1

        self.avg = self.sum / self.count

    def value(self):
        # ap for each class [num_classes]
        val = self.val
        avg = self.avg
        sum = self.sum
        count = self.count
        head_avg = avg[self.head]
        medium_avg = avg[self.medium]
        tail_avg = avg[self.tail]

        head_avg = head_avg[torch.where(~(torch.isnan(head_avg)))[0]].mean()
        medium_avg = medium_avg[torch.where(~(torch.isnan(medium_avg)))[0]].mean()
        tail_avg = tail_avg[torch.where(~(torch.isnan(tail_avg)))[0]].mean()

        return {"tail": tail_avg, "medium": medium_avg, "head": head_avg}
class LTconfMeter(object):
    def __init__(self, indices):
        super(LTconfMeter, self).__init__()
        self.tail, self.medium, self.head = indices
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

        # confidence for only correct samples
        self.cval = torch.zeros(200).float()
        self.cavg = torch.zeros(200).float()
        self.csum = torch.zeros(200).float()
        self.ccount = torch.zeros(200).float()

    def reset(self):
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

        self.cval = torch.zeros(200).float()
        self.cavg = torch.zeros(200).float()
        self.csum = torch.zeros(200).float()
        self.ccount = torch.zeros(200).float()

    def update(self, val, target, n=1):
        self.valmean = val.mean(0).clone().detach().cpu()
        # print(self.val.shape)
        self.sum += self.valmean
        self.count += val.size(0)
        self.avg = self.sum / self.count

        for idx, ml in enumerate(target):

            self.cval[int(ml)] = val[idx][int(ml)]
            self.csum[int(ml)] += val[idx][int(ml)]
            self.ccount[int(ml)] += 1

        self.cavg = self.csum / self.ccount

    def value(self):
        # ap for each class [num_classes]
        val = self.val
        avg = self.avg
        sum = self.sum
        count = self.count
        cavg = self.cavg
        head_avg = avg[self.head]
        medium_avg = avg[self.medium]
        tail_avg = avg[self.tail]

        head_cavg = cavg[self.head]
        medium_cavg = cavg[self.medium]
        tail_cavg = cavg[self.tail]

        head_avg = head_avg[torch.where(~(torch.isnan(head_avg)))[0]].mean()
        medium_avg = medium_avg[torch.where(~(torch.isnan(medium_avg)))[0]].mean()
        tail_avg = tail_avg[torch.where(~(torch.isnan(tail_avg)))[0]].mean()

        head_cavg = head_cavg[torch.where(~(torch.isnan(head_cavg)))[0]].mean()
        medium_cavg = medium_cavg[torch.where(~(torch.isnan(medium_cavg)))[0]].mean()
        tail_cavg = tail_cavg[torch.where(~(torch.isnan(tail_cavg)))[0]].mean()

        return {"tail": tail_avg, "medium": medium_avg, "head": head_avg, "tail_c": tail_cavg, "medium_c": medium_cavg, "head_c": head_cavg}

class LTEviMeter(object):
    def __init__(self, indices):
        super(LTEviMeter, self).__init__()
        self.tail, self.medium, self.head = indices
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

    def reset(self):
        self.val = torch.zeros(200).float()
        self.avg = torch.zeros(200).float()
        self.sum = torch.zeros(200).float()
        self.count = torch.zeros(200).float()

    def update(self, val, target, n=1):
        for idx, ml in enumerate(target):

            self.val[int(ml)] = val[idx]
            self.sum[int(ml)] += val[idx]
            self.count[int(ml)] += 1

        self.avg = self.sum / self.count

    def value(self):
        # ap for each class [num_classes]
        val = self.val
        avg = self.avg
        sum = self.sum
        count = self.count

        head_avg = avg[self.head]
        medium_avg = avg[self.medium]
        tail_avg = avg[self.tail]

        head_avg = head_avg[torch.where(~(torch.isnan(head_avg)))[0]].mean()
        medium_avg = medium_avg[torch.where(~(torch.isnan(medium_avg)))[0]].mean()
        tail_avg = tail_avg[torch.where(~(torch.isnan(tail_avg)))[0]].mean()

        return {"tail": tail_avg, "medium": medium_avg, "head": head_avg}

def single_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
       For single class labels
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target,  topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       For multi-class labels
    """
    # print(output.shape, target.shape)
    with torch.no_grad():
        res = []
        for k in topk:
            maxk = k
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            onehot_pred = torch.zeros(target.size())

            for i in range(batch_size):
                onehot_pred[i][pred[i]] = 1

            correct_map = onehot_pred.float().cuda() * target

            correct = torch.nonzero(correct_map.sum(1)).size(0)

            res.append(correct * 100.0 / batch_size)
        return res

def perclsaccuracy(output, target,  topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       For multi-class labels
    """
    # print(output.shape, target.shape)
    with torch.no_grad():
        res = []
        for k in topk:
            maxk = k
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            onehot_pred = torch.zeros(target.size())

            for i in range(batch_size):
                onehot_pred[i][pred[i]] = 1

            correct_map = onehot_pred.float().cuda() * target
            correct_map = correct_map.sum(1)
            correct_map[correct_map >= 1] = 1

            res.append(correct_map)

        return res

