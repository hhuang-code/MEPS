import os
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import args
from shapenet_data import ShapeNet
from shapenet_model import OuterNet, InnerNet

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


def knn(x, k):
    # input x: (b, n_pts, ch=3)
    x = x.transpose(1, 2)  # (b, ch, n_pts)

    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (b, n_pts, k)

    idx = idx + 1  # NOTE: valid vertex index in adj starting with 1

    return idx


def output_color_point_cloud(data, seg, color_map, out_file):
    # colors = []
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            # colors.append(color)
    # plot_pc(data, np.array(colors), name=out_file.split('.')[0] + '.png')


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


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
    if args.dataset == 'shapenet':
        logging.basicConfig(filename=os.path.join(args.log_path, args.dataset + args.log_name), filemode='a',
                            level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        dataloader = DataLoader(ShapeNet(args.dataset_path, args.num_points, 'train'),
                                args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=4)
        dataloader_test = DataLoader(ShapeNet(args.dataset_path, args.num_points, mode='test'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4)

        oid2cpid = json.load(open(os.path.join(args.dataset_path, 'overallid_to_catid_partid.json'), 'r'))

        # key: category; value: list of part index in this category
        object2setofoid = {}
        for idx in range(len(oid2cpid)):  # for each part in all shapes of all categories
            objid, pid = oid2cpid[idx]
            if not objid in object2setofoid.keys():
                object2setofoid[objid] = []
            object2setofoid[objid].append(idx)

        # map category name to category id, e.g., ('Airplane', '02691156')
        all_obj_cats_file = os.path.join(args.dataset_path, 'all_object_categories.txt')
        fin = open(all_obj_cats_file, 'r')
        lines = [line.rstrip() for line in fin.readlines()]
        objcats = [line.split()[1] for line in lines]
        # all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
        fin.close()

        # segment id in each category
        all_cats = json.load(open(os.path.join(args.dataset_path, 'overallid_to_catid_partid.json'), 'r'))
        args.num_part_cats = len(all_cats)

        color_map_file = os.path.join(args.dataset_path, 'part_color_mapping.json')
        color_map = json.load(open(color_map_file, 'r'))

        outer_net = OuterNet(arch_file=args.arch_file, in_channels=args.num_input_channels)
        inner_net = InnerNet(in_channels=args.num_input_channels, num_classes=args.num_classes)

        outer_net.cuda()
        inner_net.cuda()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(list(outer_net.parameters()) + list(inner_net.parameters()),
                               lr=args.learning_rate, weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

        if args.resume_model_name is not None and args.resume_model_name != '':
            checkpoint = torch.load(os.path.join(args.model_path, args.resume_model_name))
            outer_net.load_state_dict(checkpoint['outer_state_dict'])
            inner_net.load_state_dict(checkpoint['inner_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1

        if torch.cuda.device_count() > 1:
            outer_net = nn.DataParallel(outer_net)
            inner_net = nn.DataParallel(inner_net)

        max_epoch = args.max_epoch

        best_test_acc = 0
        best_test_iou = 0

        train_adj = dict()
        test_adj = dict()

        for epoch in range(start_epoch, max_epoch + 1):

            train_total_loss = 0
            # val_total_loss = 0
            train_avg_acc = AverageMeter()
            # val_avg_acc = AverageMeter()

            outer_net.train()
            inner_net.train()
            for i, input in enumerate(dataloader):
                id, x, category, label = input['id'], input['x'].cuda(), input['category'], input['label'].cuda()

                adj = []
                for j, idx in enumerate(id):
                    if idx.item() in train_adj.items():
                        adj_idx = train_adj.get(idx.item()).cuda()
                    else:
                        adj_idx = knn(x[j].unsqueeze(0), args.max_neigh)
                        train_adj[idx.item()] = adj_idx.cpu()
                    adj.append(adj_idx)
                adj = torch.cat(adj, dim=0)

                pred_weight = outer_net(x, adj)
                y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                optimizer.zero_grad()
                loss = criterion(y.contiguous().view(-1, args.num_classes), label.view(-1).long())
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()

                [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
                train_avg_acc.update(acc.item(), len(label.view(-1)))

            if optimizer.param_groups[0]['lr'] * args.lr_decay_rate >= args.lr_clip:
                scheduler.step()

            total_acc = 0.0
            total_seen = 0
            total_acc_iou = 0.0

            total_per_cat_acc = np.zeros((args.num_categories)).astype(np.float32)
            total_per_cat_iou = np.zeros((args.num_categories)).astype(np.float32)
            total_per_cat_seen = np.zeros((args.num_categories)).astype(np.int32)

            with (torch.no_grad()):
                outer_net.eval()
                inner_net.eval()
                for i, input in enumerate(dataloader_test):
                    # id, x, category, label, pts = input['id'], \
                    #     input['x'].cuda(), input['category'].numpy()[0], input['label'].numpy(), input['pts'].numpy()

                    id, x, category, label = input['id'], input['x'].cuda(), input['category'], input['label'].numpy()

                    adj = []
                    for j, idx in enumerate(id):
                        if idx.item() in test_adj.items():
                            adj_idx = test_adj.get(idx.item()).cuda()
                        else:
                            adj_idx = knn(x[j].unsqueeze(0), args.max_neigh)
                            test_adj[idx.item()] = adj_idx.cpu()
                        adj.append(adj_idx)
                    adj = torch.cat(adj, dim=0)

                    pred_weight = outer_net(x, adj)
                    seg_pred_res = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                    ori_point_num = label.shape[1]

                    iou_oids = object2setofoid[objcats[category]]
                    non_cat_labels = list(set(np.arange(args.num_classes)).difference(set(iou_oids)))

                    seg_pred_res = seg_pred_res[0].cpu().numpy()

                    mini = np.min(seg_pred_res)
                    seg_pred_res[:, non_cat_labels] = mini - 1000

                    seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]

                    seg_acc = np.mean(seg_pred_val == label)

                    total_acc += seg_acc
                    total_seen += 1

                    total_per_cat_seen[category] += 1
                    total_per_cat_acc[category] += seg_acc

                    mask = np.int32(seg_pred_val == label)

                    total_iou = 0.0
                    iou_log = ''
                    for oid in iou_oids:
                        n_pred = np.sum(seg_pred_val == oid)
                        n_gt = np.sum(label == oid)
                        n_intersect = np.sum(np.int32(label == oid) * mask)
                        n_union = n_pred + n_gt - n_intersect
                        iou_log += '_' + str(n_pred) + '_' + str(n_gt) + '_' + str(n_intersect) + '_' + str(n_union) + '_'
                        if n_union == 0:
                            total_iou += 1
                            iou_log += '_1\n'
                        else:
                            total_iou += n_intersect * 1.0 / n_union
                            iou_log += '_' + str(n_intersect * 1.0 / n_union) + '\n'

                    avg_iou = total_iou / len(iou_oids)
                    total_acc_iou += avg_iou
                    total_per_cat_iou[category] += avg_iou

                    if args.visualize:
                        pts = pts.squeeze(0)
                        label = label.squeeze(0)
                        output_color_point_cloud(pts, label, color_map,
                                                 os.path.join(args.result_path, str(i) + '_gt.obj'))
                        output_color_point_cloud(pts, seg_pred_val, color_map,
                                                 os.path.join(args.result_path, str(i) + '_pred.obj'))
                        output_color_point_cloud_red_blue(pts, np.int32(label == seg_pred_val),
                                                          os.path.join(args.result_path, str(i) + '_diff.obj'))

                msg = 'Accuracy: {}'.format(total_acc / total_seen)
                print(msg)
                logging.info(msg)

                msg = 'IoU: {}'.format(total_acc_iou / total_seen)
                print(msg)
                logging.info(msg)

                for cat_idx in range(args.num_categories):
                    msg = '\t ' + objcats[cat_idx] + ' Total Number: ' + str(total_per_cat_seen[cat_idx])
                    print(msg)
                    logging.info(msg)

                    if total_per_cat_seen[cat_idx] > 0:
                        msg = '\t ' + objcats[cat_idx] + ' Accuracy: ' + \
                              str(total_per_cat_acc[cat_idx] / total_per_cat_seen[cat_idx])
                        print(msg)
                        logging.info(msg)

                        msg = '\t ' + objcats[cat_idx] + ' IoU: ' + \
                              str(total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx])
                        print(msg)
                        logging.info(msg)

                if best_test_iou < total_acc_iou / total_seen:
                    best_test_iou = total_acc_iou / total_seen
                    best_test_acc = total_acc / total_seen

                    if isinstance(inner_net, nn.DataParallel):
                        inner_state_dict = inner_net.module.state_dict()
                        outer_state_dict = outer_net.module.state_dict()
                    else:
                        inner_state_dict = inner_net.state_dict()
                        outer_state_dict = outer_net.state_dict()
                    optimizer_state_dict = optimizer.state_dict()
                    scheduler_state_dict = scheduler.state_dict()
                    torch.save({'epoch': epoch,
                                'inner_state_dict': inner_state_dict, 'outer_state_dict': outer_state_dict,
                                'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict,
                                'best_test_accuracy': best_test_acc, 'best_test_iou': best_test_iou},
                               '{0}/{1}'.format(args.model_path, 'best_' + args.model_name))

            msg = 'Epoch {}/{}: avg train loss: {:5f}, avg train accuracy: {:3f}, ' \
                  'avg test accuracy: {:3f}, avg test iou: {:3f}, best test accuracy: {:3f}, best test iou: {:3f}'.format(
                epoch, max_epoch, train_total_loss / len(dataloader), train_avg_acc.avg,
                                  total_acc / total_seen, total_acc_iou / total_seen, best_test_acc, best_test_iou)

            print(msg)
            logging.info(msg)

            # with torch.no_grad():
            #     for i, input in enumerate(dataloader_val):
            #         id, x, category, label = input['id'], input['x'].cuda(), input['category'], input['label'].cuda()
            #
            #         adj = []
            #         for j, idx in enumerate(id):
            #             if idx.item() in val_adj.items():
            #                 adj_idx = val_adj.get(idx.item()).cuda()
            #             else:
            #                 adj_idx = knn(x[j].unsqueeze(0), args.max_neigh)
            #                 val_adj[idx.item()] = adj_idx.cpu()
            #             adj.append(adj_idx)
            #         adj = torch.cat(adj, dim=0)
            #
            #         pred_weight = outer_net(x, adj)
            #         y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)
            #
            #         loss = criterion(y.contiguous().view(-1, args.num_classes), label.view(-1).long())
            #
            #         val_total_loss += loss.item()
            #
            #         [acc] = accuracy(y.contiguous().view(-1, args.num_classes), label.view(-1), topk=(1,))
            #         val_avg_acc.update(acc.item(), len(label.view(-1)))
            #
            #     if best_val_acc < val_avg_acc.avg:
            #         best_val_acc = val_avg_acc.avg
            #
            #         if isinstance(inner_net, nn.DataParallel):
            #             inner_state_dict = inner_net.module.state_dict()
            #             outer_state_dict = outer_net.module.state_dict()
            #         else:
            #             inner_state_dict = inner_net.state_dict()
            #             outer_state_dict = outer_net.state_dict()
            #         torch.save({'epoch': epoch, 'inner_state_dict': inner_state_dict,
            #                     'outer_state_dict': outer_state_dict, 'best_val_accuracy': best_val_acc},
            #                    '{0}/{1}'.format(args.model_path, 'shapenet_best.pth.tar'))
            #
            # msg = 'Epoch {}/{}: avg train loss: {:5f}, avg train accuracy: {:3f}, ' \
            #       'avg val loss: {:5f}, avg val accuracy: {:3f}, best test accuracy: {:3f}'.format(
            #     epoch, max_epoch, train_total_loss / len(dataloader), train_avg_acc.avg,
            #                       val_total_loss / len(dataloader_val), val_avg_acc.avg, best_val_acc)
            #
            # print(msg)
            # logging.info(msg)


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
