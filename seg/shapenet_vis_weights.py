import json
import logging
import os
import pdb

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from config import args
from shapenet_data import ShapeNet
from shapenet_model import OuterNet, InnerNet

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True


def knn(x, k):
    # input x: (b, n_pts, ch=3)
    x = x.transpose(1, 2)  # (b, ch, n_pts)

    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (b, n_pts, k)

    idx = idx + 1  # NOTE: valid vertex index in adj starting with 1

    return idx


# def output_color_point_cloud(data, seg, color_map, out_file):
def output_color_point_cloud(data, color_map, out_file):
    # colors = []
    with open(out_file, 'w') as f:
        # l = len(seg)
        l = len(data)
        for i in range(l):
            # color = color_map[seg[i]]
            color = color_map[i]
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

        args.log_name = '_test.log'

        logging.basicConfig(filename=os.path.join(args.log_path, args.dataset + args.log_name), filemode='a',
                            level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        # dataloader = DataLoader(ShapeNet(args.dataset_path, args.num_points, 'train'),
        #                         args.batch_size,
        #                         shuffle=True,
        #                         drop_last=True,
        #                         num_workers=4)
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
        objcatnames = [line.split()[0] for line in lines]
        obj_cat2name = {k:v for k, v in zip(objcats, objcatnames)}
        # all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
        fin.close()

        # segment id in each category
        all_cats = json.load(open(os.path.join(args.dataset_path, 'overallid_to_catid_partid.json'), 'r'))
        args.num_part_cats = len(all_cats)

        color_map_file = os.path.join(args.dataset_path, 'part_color_mapping.json')
        color_map = json.load(open(color_map_file, 'r'))

        outer_net = OuterNet(arch_file=args.arch_file, in_channels=args.num_input_channels)
        inner_net = InnerNet(in_channels=args.num_input_channels, num_classes=args.num_classes)

        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))
        outer_net.load_state_dict(checkpoint['outer_state_dict'])
        inner_net.load_state_dict(checkpoint['inner_state_dict'])

        best_test_accuracy = checkpoint['best_test_accuracy']
        best_test_iou = checkpoint['best_test_iou']
        print(f'best_test_accuracy: {best_test_accuracy}\nbest_test_iou: {best_test_iou}')

        outer_net.cuda()
        inner_net.cuda()

        if torch.cuda.device_count() > 1:
            outer_net = nn.DataParallel(outer_net)
            inner_net = nn.DataParallel(inner_net)

        # criterion = nn.CrossEntropyLoss()
        #
        # optimizer = optim.Adam(list(outer_net.parameters()) + list(inner_net.parameters()),
        #                        lr=args.learning_rate, weight_decay=args.weight_decay)
        #
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
        #
        # max_epoch = args.max_epoch

        test_adj = dict()

        total_acc = 0.0
        total_seen = 0
        total_acc_iou = 0.0

        total_per_cat_acc = np.zeros((args.num_categories)).astype(np.float32)
        total_per_cat_iou = np.zeros((args.num_categories)).astype(np.float32)
        total_per_cat_seen = np.zeros((args.num_categories)).astype(np.int32)


        cat_cnt = dict()


        with (torch.no_grad()):
            outer_net.eval()
            inner_net.eval()
            for i, input in enumerate(dataloader_test):

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
                # seg_pred_res = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

                pts = x.squeeze(0).cpu().numpy()

                cat_name = obj_cat2name[objcats[category.item()]]

                if cat_name not in cat_cnt.keys():
                    cat_cnt[cat_name] = 1
                else:
                    cat_cnt[cat_name] += 1

                if cat_cnt[cat_name] <= 5:
                    _all = []
                    keys = list(pred_weight.keys())
                    for _, (key, value) in enumerate(zip(keys, pred_weight.values())):
                        conv = []
                        for j, param in enumerate(value):
                            weight, bias = param['weight'], param['bias']

                            weight = weight.detach().cpu().numpy().squeeze(0).reshape(weight.shape[1], -1)
                            weight_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(weight).squeeze(-1)  # (n_pts,)

                            bias = bias.detach().cpu().numpy().squeeze(0)
                            bias_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(bias).squeeze(-1)  # (n_pts,)

                            weight_norm = mpl.colors.Normalize(vmin=weight_tsne.min(), vmax=weight_tsne.max())
                            weight_cmap = cm.rainbow(weight_norm(weight_tsne)) # (6890, 4)

                            output_color_point_cloud(pts, weight_cmap, os.path.join(args.visual_path, 'weights', cat_name + '_' + str(cat_cnt[cat_name]) + '_' + key + '_' + str(j + 1) + '_weight.obj'))

                            bias_norm = mpl.colors.Normalize(vmin=bias_tsne.min(), vmax=bias_tsne.max())
                            bias_cmap = cm.rainbow(bias_norm(bias_tsne))  # (6890, 4)

                            output_color_point_cloud(pts, bias_cmap, os.path.join(args.visual_path, 'weights', cat_name + '_' + str(cat_cnt[cat_name]) + '_' + key + '_' + str(j + 1) + '_bias.obj'))

                            layer = np.concatenate([weight, bias], axis=-1)

                            layer_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(layer).squeeze(-1)  # (n_pts,)

                            layer_norm = mpl.colors.Normalize(vmin=layer_tsne.min(), vmax=layer_tsne.max())
                            layer_cmap = cm.rainbow(layer_norm(layer_tsne))  # (6890, 4)

                            output_color_point_cloud(pts, layer_cmap, os.path.join(args.visual_path, 'weights', cat_name + '_' + str(cat_cnt[cat_name]) + '_' + key + '_' + str(j + 1) + '_layer.obj'))

                            conv.append(layer)

                        conv = np.concatenate(conv, axis=-1)

                        conv_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(conv).squeeze(-1)  # (n_pts,)

                        conv_norm = mpl.colors.Normalize(vmin=conv_tsne.min(), vmax=conv_tsne.max())
                        conv_cmap = cm.rainbow(conv_norm(conv_tsne))  # (6890, 4)

                        output_color_point_cloud(pts, conv_cmap, os.path.join(args.visual_path, 'weights', cat_name + '_' + str(cat_cnt[cat_name]) + '_' + key + '_conv.obj'))

                        _all.append(conv)

                    _all = np.concatenate(_all, axis=-1)

                    all_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(_all).squeeze(-1)  # (n_pts,)

                    all_norm = mpl.colors.Normalize(vmin=all_tsne.min(), vmax=all_tsne.max())
                    all_cmap = cm.rainbow(all_norm(all_tsne))  # (6890, 4)

                    output_color_point_cloud(pts, all_cmap, os.path.join(args.visual_path, 'weights', cat_name + '_' + str(cat_cnt[cat_name]) + '_all.obj'))

                # ori_point_num = label.shape[1]
                #
                # iou_oids = object2setofoid[objcats[category]]
                # non_cat_labels = list(set(np.arange(args.num_classes)).difference(set(iou_oids)))
                #
                # seg_pred_res = seg_pred_res[0].cpu().numpy()
                #
                # mini = np.min(seg_pred_res)
                # seg_pred_res[:, non_cat_labels] = mini - 1000
                #
                # seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]
                #
                # seg_acc = np.mean(seg_pred_val == label)
                #
                # total_acc += seg_acc
                # total_seen += 1
                #
                # total_per_cat_seen[category] += 1
                # total_per_cat_acc[category] += seg_acc
                #
                # mask = np.int32(seg_pred_val == label)
                #
                # total_iou = 0.0
                # iou_log = ''
                # for oid in iou_oids:
                #     n_pred = np.sum(seg_pred_val == oid)
                #     n_gt = np.sum(label == oid)
                #     n_intersect = np.sum(np.int32(label == oid) * mask)
                #     n_union = n_pred + n_gt - n_intersect
                #     iou_log += '_' + str(n_pred) + '_' + str(n_gt) + '_' + str(n_intersect) + '_' + str(
                #         n_union) + '_'
                #     if n_union == 0:
                #         total_iou += 1
                #         iou_log += '_1\n'
                #     else:
                #         total_iou += n_intersect * 1.0 / n_union
                #         iou_log += '_' + str(n_intersect * 1.0 / n_union) + '\n'
                #
                # avg_iou = total_iou / len(iou_oids)
                # total_acc_iou += avg_iou
                # total_per_cat_iou[category] += avg_iou
                #
                # if args.visualize:
                #     pts = x.squeeze(0).cpu().numpy()
                #     label = label.squeeze(0)
                #     output_color_point_cloud(pts, label, color_map,
                #                              os.path.join(args.visual_path, obj_cat2name[objcats[category.item()]] + '_' + str(i) + '_gt.obj'))
                #     output_color_point_cloud(pts, seg_pred_val, color_map,
                #                              os.path.join(args.visual_path, obj_cat2name[objcats[category.item()]] + '_' + str(i) + '_pred.obj'))
                #     output_color_point_cloud_red_blue(pts, np.int32(label == seg_pred_val),
                #                                       os.path.join(args.visual_path, obj_cat2name[objcats[category.item()]] + '_' + str(i) + '_diff.obj'))

        #     msg = 'Accuracy: {}'.format(total_acc / total_seen)
        #     print(msg)
        #     logging.info(msg)
        #
        #     msg = 'IoU: {}'.format(total_acc_iou / total_seen)
        #     print(msg)
        #     logging.info(msg)
        #
        #     for cat_idx in range(args.num_categories):
        #         msg = '\t ' + obj_cat2name[objcats[cat_idx]] + ' Total Number: ' + str(total_per_cat_seen[cat_idx])
        #         print(msg)
        #         logging.info(msg)
        #
        #         if total_per_cat_seen[cat_idx] > 0:
        #             msg = '\t ' + obj_cat2name[objcats[cat_idx]] + ' Accuracy: ' + \
        #                   str(total_per_cat_acc[cat_idx] / total_per_cat_seen[cat_idx])
        #             print(msg)
        #             logging.info(msg)
        #
        #             msg = '\t ' + obj_cat2name[objcats[cat_idx]] + ' IoU: ' + \
        #                   str(total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx])
        #             print(msg)
        #             logging.info(msg)
        #
        # msg = 'avg test accuracy: {:3f}, avg test iou: {:3f}'.format(total_acc / total_seen, total_acc_iou / total_seen)
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

    # visualize setting
    if args.visualize:
        if not os.path.exists(os.path.join(args.visual_path, 'weights')):
            os.makedirs(os.path.join(args.visual_path, 'weights'))

    LOG_FORMAT = '%(asctime)s - %(message)s'
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    run(args)
