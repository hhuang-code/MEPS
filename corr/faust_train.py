import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
random.seed(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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


def run(args):
    if args.dataset == 'faust':
        logging.basicConfig(filename=os.path.join(args.log_path, args.dataset + '.log'), filemode='a',
                            level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        dataloader = DataLoader(FAUST(args.dataset_path, args.max_neigh, 'training'),
                                args.batch_size,
                                shuffle=True,
                                num_workers=4)
        dataloader_test = DataLoader(FAUST(args.dataset_path, args.max_neigh, 'test'),
                                     args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        outer_net = OuterNet(arch_file=args.arch_file, in_channels=args.num_input_channels)
        outer_net.cuda()

        inner_net = InnerNet(in_channels=args.num_input_channels, num_classes=args.num_classes)
        inner_net.cuda()

        if torch.cuda.device_count() > 1:
            outer_net = nn.DataParallel(outer_net)
            inner_net = nn.DataParallel(inner_net)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(list(outer_net.parameters()) + list(inner_net.parameters()),
                               lr=args.learning_rate, weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

        max_epoch = args.num_iterations // args.batch_size + 1
        best_test_acc = 0

        for epoch in range(1, max_epoch + 1):
            train_total_loss = 0
            test_total_loss = 0
            train_avg_acc = AverageMeter()
            test_avg_acc = AverageMeter()

            outer_net.train()
            inner_net.train()
            for i, input in enumerate(dataloader):
                x, adj = input['x'].cuda(), input['adj'].cuda()  # NOTE: valid vertex index in adj starting with 1

                pred_weight = outer_net(x, adj)
                y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                cur_batch_size = x.shape[0]
                label = torch.arange(0, args.num_classes).cuda()
                label = label.unsqueeze(0).repeat(cur_batch_size, 1)

                optimizer.zero_grad()
                loss = criterion(y.contiguous().view(-1, args.num_classes), label.view(-1))
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()

                [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
                train_avg_acc.update(acc.item(), len(label.view(-1)))

            if optimizer.param_groups[0]['lr'] * args.lr_decay_rate >= args.lr_clip:
                scheduler.step()

            outer_net.eval()
            inner_net.eval()
            with torch.no_grad():
                for i, input in enumerate(dataloader_test):
                    x, adj = input['x'].cuda(), input['adj'].cuda()  # NOTE: valid vertex index in adj starting with 1

                    pred_weight = outer_net(x, adj)
                    y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                    cur_batch_size = x.shape[0]
                    label = torch.arange(0, args.num_classes).cuda()
                    label = label.unsqueeze(0).repeat(cur_batch_size, 1)

                    loss = criterion(y.contiguous().view(-1, args.num_classes), label.view(-1))

                    test_total_loss += loss.item()

                    [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
                    test_avg_acc.update(acc.item(), len(label.view(-1)))

                if best_test_acc < test_avg_acc.avg:
                    best_test_acc = test_avg_acc.avg

                    if isinstance(inner_net, nn.DataParallel):
                        inner_state_dict = inner_net.module.state_dict()
                        outer_state_dict = outer_net.module.state_dict()
                    else:
                        inner_state_dict = inner_net.state_dict()
                        outer_state_dict = outer_net.state_dict()
                    torch.save({'epoch': epoch, 'inner_state_dict': inner_state_dict,
                                'outer_state_dict': outer_state_dict, 'best_test_accuracy': best_test_acc},
                               '{0}/{1}'.format(args.model_path, 'faust_best.pth.tar'))

            msg = 'Epoch {}/{}: avg train loss: {:5f}, avg train accuracy: {:3f}, ' \
                  'avg test loss: {:5f}, avg test accuracy: {:3f}, best test accuracy: {:3f}'.format(
                epoch, max_epoch, train_total_loss / len(dataloader), train_avg_acc.avg,
                                  test_total_loss / len(dataloader_test), test_avg_acc.avg, best_test_acc)

            print(msg)
            logging.info(msg)


if __name__ == '__main__':
    # checkpoing
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
