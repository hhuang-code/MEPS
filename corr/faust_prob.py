# -----------------------------------------------------------
# 1. before refinement, save predicted class probability (not label itself) for vertex labels in a prob.mat file
# 2. after refinement using a Matlab program, compute correspondence accuracy by loading result_LAP.mat file
# -----------------------------------------------------------

import os
import time
import logging
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from config import args
from faust_data import FAUST
from faust_model import OuterNet, InnerNet

import pdb

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """ computes the precision@k for the specified values of k """
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


class AverageMeter(object):
    """ computes and stores the average and current value """

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):
    if os.path.exists(os.path.join(args.result_path, 'result_LAP.mat')):
        y = sio.loadmat(os.path.join(args.result_path, 'result_LAP.mat'))['pred']
        n_samples, n_pts = y.shape
        gt = np.arange(1, n_pts + 1)
        gt = np.repeat(np.expand_dims(gt, axis=0), n_samples, axis=0)
        acc = (y == gt).sum() / (n_samples * n_pts)
        print('avg test accuracy after LAP: {:3f}'.format(acc))
    else:
        logging.basicConfig(filename=os.path.join(args.log_path, args.dataset + '-test.log'), filemode='a',
                            level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        dataloader_test = DataLoader(FAUST(args.dataset_path, args.max_neigh, 'test'),
                                     1, #args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        outer_net = OuterNet(arch_file=args.arch_file, in_channels=args.num_input_channels)
        inner_net = InnerNet(in_channels=args.num_input_channels, num_classes=args.num_classes)

        print(f'Number of parameters: {count_parameters(outer_net) + count_parameters(inner_net)}')

        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))
        outer_net.load_state_dict(checkpoint['outer_state_dict'])
        inner_net.load_state_dict(checkpoint['inner_state_dict'])

        outer_net.cuda()
        inner_net.cuda()

        if torch.cuda.device_count() > 1:
            outer_net = nn.DataParallel(outer_net)
            inner_net = nn.DataParallel(inner_net)

        test_avg_acc = AverageMeter()
        prob_list, pred_list, gt_list = [], [], []

        times = []

        outer_net.eval()
        inner_net.eval()
        with torch.no_grad():

            for i, data in enumerate(dataloader_test):
                x, adj = data['x'].cuda(), data['adj'].cuda()  # NOTE: valid vertex index in adj starting with 1

                st_time = time.time()

                pred_weight = outer_net(x, adj)
                y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                ed_time = time.time()
                times.append(ed_time - st_time)

                cur_batch_size = x.shape[0]
                label = torch.arange(0, args.num_classes).cuda()
                label = label.unsqueeze(0).repeat(cur_batch_size, 1)

                [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
                test_avg_acc.update(acc.item(), len(label.view(-1)))

                pred = y.topk(1, -1, True, True)[1].squeeze(-1)
                gt = label

                prob_list.append(y)
                pred_list.append(pred)
                gt_list.append(gt)

        prob_list = torch.cat(prob_list, dim=0).cpu().numpy()
        pred_list = torch.cat(pred_list, dim=0).cpu().numpy()
        gt_list = torch.cat(gt_list, dim=0).cpu().numpy()
        print(
            'acc: {:3f}'.format(float(np.count_nonzero(pred_list == gt_list)) / (gt_list.shape[0] * gt_list.shape[1])))

        msg = 'avg test accuracy: {:3f}'.format(test_avg_acc.avg)

        print(msg)
        logging.info(msg)

        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

        sio.savemat(os.path.join(args.result_path, 'prob.mat'), {'prob': prob_list})

        print(f'Time: {sum(times) / float(len(times))}')


if __name__ == '__main__':
    # checkpoint
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # log setting
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    LOG_FORMAT = '%(asctime)s - %(message)s'
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    run(args)
