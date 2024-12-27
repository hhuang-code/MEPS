import os
import glob
import logging
import threading
import numpy as np
from time import sleep
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def area_under_curve(points):
    area = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        area += (x2 - x1) * (y1 + y2) / 2

    area /= (100 * 0.1)

    return area


def run(args):
    logging.basicConfig(filename=os.path.join(args.log_path, args.dataset + '-curve.log'), filemode='a',
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    dataloader_test = DataLoader(FAUST(args.dataset_path, args.max_neigh, 'test'),
                                 args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    outer_net = OuterNet(arch_file=args.arch_file, in_channels=args.num_input_channels)
    inner_net = InnerNet(in_channels=args.num_input_channels, num_classes=args.num_classes)

    checkpoint = torch.load(os.path.join(args.model_path, args.model_name))
    outer_net.load_state_dict(checkpoint['outer_state_dict'])
    inner_net.load_state_dict(checkpoint['inner_state_dict'])

    outer_net.cuda()
    inner_net.cuda()

    if torch.cuda.device_count() > 1:
        outer_net = nn.DataParallel(outer_net)
        inner_net = nn.DataParallel(inner_net)

    test_avg_acc = AverageMeter()
    pred_list, gt_list = [], []

    outer_net.eval()
    inner_net.eval()
    with torch.no_grad():

        for i, data in enumerate(dataloader_test):
            x, adj = data['x'].cuda(), data['adj'].cuda()  # NOTE: valid vertex index in adj starting with 1

            pred_weight = outer_net(x, adj)
            y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

            cur_batch_size = x.shape[0]
            label = torch.arange(0, args.num_classes).cuda()
            label = label.unsqueeze(0).repeat(cur_batch_size, 1)

            [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
            test_avg_acc.update(acc.item(), len(label.view(-1)))

            pred = y.topk(1, -1, True, True)[1].squeeze(-1)
            gt = label

            pred_list.append(pred)
            gt_list.append(gt)

    pred_list = torch.cat(pred_list, dim=0).cpu().numpy()
    gt_list = torch.cat(gt_list, dim=0).cpu().numpy()
    print('acc: {:3f}'.format(float(np.count_nonzero(pred_list == gt_list)) / (gt_list.shape[0] * gt_list.shape[1])))

    msg = 'avg test accuracy: {:3f}'.format(test_avg_acc.avg)

    print(msg)
    logging.info(msg)

    # ==================================================
    sleep(3)  # take a break

    # load refined prediction
    pred_refine_list = sio.loadmat(os.path.join(args.result_path, 'result_LAP.mat'))['pred']  # (20, 6890)
    n_samples, n_pts = pred_refine_list.shape
    gt = np.arange(1, n_pts + 1)
    gt = np.repeat(np.expand_dims(gt, axis=0), n_samples, axis=0)
    acc = (pred_refine_list == gt).sum() / (n_samples * n_pts)
    print('avg test accuracy after LAP: {:3f}'.format(acc))

    pred_refine_list -= 1  # label starting from 0

    off_path = './MPI-FAUST/registrations/*.off'
    off_list = sorted(glob.glob(off_path))
    test_off = off_list[80:]

    root_dir = args.result_path
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        re_run_pred = 'y'
        re_run_refined = 'y'
    else:
        re_run_pred = input('Re-run for geodesic distance of PRED [y/n]: ')
        re_run_refined = input('Re-run for geodesic distance of REFINED [y/n]: ')
        if re_run_pred == 'y':
            os.system('rm -f {}/pred_[0-9]*.txt'.format(root_dir))
            os.system('rm -f {}/distances_[0-9]*.txt'.format(root_dir))
            os.system('rm -f {}/faust_distance_ours.npy'.format(root_dir))
        if re_run_refined == 'y':
            os.system('rm -f {}/pred_refine_[0-9]*.txt'.format(root_dir))
            os.system('rm -f {}/distances_refine_[0-9]*.txt'.format(root_dir))
            os.system('rm -f {}/faust_distance_refine_ours.npy'.format(root_dir))

    class CustomThread(threading.Thread):
        def __init__(self, name=None, file_index=None, refinement=False):
            threading.Thread.__init__(self, name=name)
            self.name = name
            self.file_index = file_index
            self.refinement = refinement

        # use refinement result
        def run(self):
            off_f = test_off[self.file_index]
            if self.refinement:
                # save prediction result
                np.savetxt(os.path.join(root_dir, 'pred_refine_{}.txt'.format(self.file_index)),
                           pred_refine_list[self.file_index].astype('int32'))
                # compute geodesic distance and save to distances.txt
                exec = './example0 ' + off_f + ' {}/pred_refine_{}.txt'.format(root_dir, self.file_index) + \
                       ' {}/distances_refine_{}.txt'.format(root_dir, self.file_index)
            else:
                # save prediction result
                np.savetxt(os.path.join(root_dir, 'pred_{}.txt'.format(self.file_index)),
                           pred_list[self.file_index].astype('int32'))
                # compute geodesic distance and save to distances.txt
                exec = './example0 ' + off_f + ' {}/pred_{}.txt'.format(root_dir, self.file_index) + \
                       ' {}/distances_{}.txt'.format(root_dir, self.file_index)
            print(exec)
            os.system(exec)

    # geodesic distance calculation
    if re_run_pred == 'y':
        thread_list = []
        for i in range(20):
            thread_list.append(CustomThread('thread_{}'.format(i), i, refinement=False))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()

    # combine all geodesic distantces.txt files into a single .npy file
    distance_all = []
    for i in range(20):
        distance_cur = np.loadtxt(os.path.join(root_dir, 'distances_{}.txt'.format(i)))  # (6891,)
        distance_all.append((i, distance_cur[:6890].reshape((1, -1))))

    distance_all.sort(key=lambda x: x[0])
    distance_all = [x[1] for x in distance_all]

    distance_cmb = np.concatenate(distance_all, axis=0)
    np.save(os.path.join(args.result_path, 'faust_distance_ours.npy'), distance_cmb)

    if re_run_refined == 'y':
        thread_list = []
        for i in range(20):
            thread_list.append(CustomThread('thread_refine_{}'.format(i), i, refinement=True))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()

    # combine all geodesic distantces.txt files into a single .npy file
    distance_refine_all = []
    for i in range(20):
        distance_cur = np.loadtxt(os.path.join(root_dir, 'distances_refine_{}.txt'.format(i)))  # (6891,)
        distance_refine_all.append((i, distance_cur[:6890].reshape((1, -1))))

    distance_refine_all.sort(key=lambda x: x[0])
    distance_refine_all = [x[1] for x in distance_refine_all]

    distance_refine_cmb = np.concatenate(distance_refine_all, axis=0)
    np.save(os.path.join(args.result_path, 'faust_distance_refine_ours.npy'), distance_refine_cmb)

    colors = plt.cm.jet(np.linspace(0, 1, 10))

    # plot our results w/o refinement
    distance_norefine = np.load(os.path.join(args.result_path, 'faust_distance_ours.npy'))
    ours_norefine = np.zeros((100))
    for i in range(100):
        ours_norefine[i] = 100 * float(np.count_nonzero(100 * distance_norefine <= 20 * i / 100.0)) / float(20 * 6890)
    plt.plot(0.1 * np.arange(100) / 100, ours_norefine, color=colors[9], linestyle='--', linewidth=2.5)
    ours_norefine = np.stack([0.1 * np.arange(100) / 100, ours_norefine], axis=1)
    print(f'Ours-NF: {area_under_curve(ours_norefine)}')

    # plot our results
    distance = np.load(os.path.join(args.result_path, 'faust_distance_refine_ours.npy'))
    ours = np.zeros((100))
    for i in range(100):
        ours[i] = 100 * float(np.count_nonzero(100 * distance <= 20 * i / 100.0)) / float(20 * 6890)
    plt.plot(0.1 * np.arange(100) / 100, ours, color=colors[9], linewidth=2.5)
    ours = np.stack([0.1 * np.arange(100) / 100, ours], axis=1)
    print(f'Ours: {area_under_curve(ours)}')

    plt.legend(['Ours-NF', 'Ours'], loc=4)

    plt.xlim([0, 0.1])
    plt.ylim(10, 100)

    plt.xlabel('Geodesic error')
    plt.ylabel('% Correspondence')
    plt.savefig(os.path.join(args.result_path, 'faust_distance.svg'), format='svg', dpi=1200)


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
