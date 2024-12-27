import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import pdb


def tile_repeat(n, rep_times):
    '''
    create something like 111..122..233..344 ..... n..nn
    One particular number appears rep_times consecutively.
    This is for flattening the indices.
    '''
    idx = torch.arange(n)
    idx = idx.view(-1, 1)  # convert to n x 1 matrix
    idx = idx.repeat(1, rep_times)
    y = idx.view(-1)

    return y


def get_patches(x, adj):
    '''
    gather neighborhood features around each point
    '''
    batch_size, num_points, in_channels = x.shape
    batch_size, num_points, K = adj.shape  # K: maximum number of neighbors
    zeros = torch.zeros(batch_size, 1, in_channels, device=x.device)
    x = torch.cat((zeros, x), dim=1)
    x = x.view(batch_size * (num_points + 1), in_channels)
    adj = adj.view(batch_size * num_points * K)
    adj_flat = tile_repeat(batch_size, num_points * K).to(adj.device)
    adj_flat = adj_flat * (num_points + 1)
    adj_flat = adj_flat + adj
    adj_flat = adj_flat.view(batch_size * num_points, K)
    patches = x[adj_flat]
    patches = patches.view(batch_size, num_points, K, in_channels)

    return patches


def get_weight_assignments(x, adj, u, v, c):
    # input x: (b, ch, n_pts), adj: (b, n_pts, K), u/v: (M, ch), c: (M)
    batch_size, _, _ = x.shape

    # (b, M, n_pts)
    ux = torch.bmm(u.unsqueeze(0).repeat(batch_size, 1, 1), x)
    vx = torch.bmm(v.unsqueeze(0).repeat(batch_size, 1, 1), x)

    # (b, n_pts, M)
    vx = vx.permute(0, 2, 1)
    # (b, n_pts, K, M)
    patches = get_patches(vx, adj)
    # (K, b, M, n_pts)
    patches = patches.permute(2, 0, 3, 1)
    # (K, b, M, n_pts)
    patches = torch.add(patches, ux)
    # (K, b, n_pts, M)
    patches = patches.permute(0, 1, 3, 2)
    # (K, b, n_pts, M)
    patches = torch.add(patches, c)
    # (b, n_pts, K, M)
    patches = patches.permute(1, 2, 0, 3)

    patches = F.softmax(patches, dim=-1)

    return patches


def get_weight_assignments_pw(x, adj, pred_weight):
    # input x: (b, ch, n_pts), adj: (b, n_pts, K)
    batch_size, ch, num_points = x.shape
    _, _, K = adj.shape

    x = x.permute(0, 2, 1)
    patches = get_patches(x, adj)  # (b, n_pts, K, ch)
    patches = patches.view(batch_size * num_points, -1, ch).transpose(1, 2)  # (b * n_pts, K, ch) -> (b * n_pts, ch, K)

    for module in pred_weight:
        _, _, out_ch, in_ch = module.get('weight').shape
        weight = module.get('weight').view(batch_size * num_points, out_ch, in_ch)
        bias = module.get('bias').view(batch_size * num_points, out_ch)
        patches = torch.bmm(weight, patches) + bias.unsqueeze(-1)
        patches = F.relu(patches)

    patches = patches.view(batch_size, num_points, -1, K).permute(0, 1, 3, 2)

    patches = F.softmax(patches, dim=-1)

    return patches


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, M_conv, translation_invariance=False):
        super(CustomConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_conv = M_conv
        self.translation_invariance = translation_invariance

        self.weight = nn.Parameter(torch.ones(M_conv, out_channels, in_channels))
        nn.init.normal_(self.weight, mean=0, std=0.1)

        self.bias = nn.Parameter(torch.ones(out_channels))
        nn.init.normal_(self.bias, mean=0, std=0.1)

        self.u = nn.Parameter(torch.ones(M_conv, in_channels))
        nn.init.normal_(self.u, mean=0, std=0.1)

        self.v = nn.Parameter(torch.ones(M_conv, in_channels))
        nn.init.normal_(self.v, mean=0, std=0.1)

        self.c = nn.Parameter(torch.ones(M_conv))
        nn.init.normal_(self.c, mean=0, std=0.1)

    def forward(self, x, adj, pred_weight=None):
        if self.translation_invariance == False:
            # input x: (b, n_pts, c), adj: (b, n_pts, n_neigh=K)
            batch_size, input_size, in_channels = x.shape
            _, _, K = adj.shape

            # calculate neighborhood size for each input
            adj_size = (adj != 0).sum(dim=-1)
            non_zeros = (adj_size != 0)
            adj_size = adj_size.float()
            adj_size = torch.where(non_zeros, torch.reciprocal(adj_size),
                                   torch.zeros_like(adj_size, device=adj_size.device))
            adj_size = adj_size.view(batch_size, input_size, 1, 1)

            x = x.permute(0, 2, 1)
            W = self.weight.view(self.m_conv * self.out_channels, self.in_channels)

            # multiple W and x -> (batch_size, M * out_channels, input_size)
            wx = torch.bmm(W.unsqueeze(0).repeat(batch_size, 1, 1), x)
            # reshape to (batch_size, input_size, M * out_channels)
            wx = wx.permute(0, 2, 1)  # (b, n_pts, M * out_ch)

            patches = get_patches(wx, adj)  # (b, n_pts, K, M * out_ch)
            # (b, n_pts, K, M)
            if pred_weight is None:
                q = get_weight_assignments(x, adj, self.u, self.v, self.c)
            else:
                q = get_weight_assignments_pw(x, adj, pred_weight)

            # element-wise multiplication of q and patches for each input
            patches = patches.view(batch_size, input_size, K, self.m_conv, self.out_channels)
            # (out_ch, b, n_pts, K, M)
            patches = patches.permute(4, 0, 1, 2, 3)
            patches = torch.mul(q, patches)
            # (b, n_pts, K, M, out_ch)
            patches = patches.permute(1, 2, 3, 4, 0)
            # add all elements for all neighbours for a particular m sum_{j in N_i} qwx -- (b, n_pts, M, out)
            patches = patches.sum(dim=2)
            # average over the number of neighbors -- (b, n_pts, M, out)
            patches = torch.mul(adj_size, patches)
            # add elements for all m -- (b, n_pts, out)
            patches = patches.sum(dim=2)
            # add bias -- (b, n_pts, out)
            patches = patches + self.bias

            return patches


class CustomLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomLinear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels))
        nn.init.normal_(self.weight, mean=0, std=0.1)

        self.bias = nn.Parameter(torch.ones(out_channels))
        nn.init.normal_(self.bias, mean=0, std=0.1)

    def forward(self, x):
        # input x: (b, n_pts, c)
        batch_size, _, _ = x.shape
        x = x.permute(0, 2, 1)
        x = torch.bmm(self.weight.unsqueeze(0).repeat(batch_size, 1, 1), x)
        x = x + self.bias.view(self.out_channels, 1)
        x = x.permute(0, 2, 1)

        return x


class InnerNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InnerNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        out_channels_fc0 = 16
        self.h_fc0 = nn.Sequential(CustomLinear(self.in_channels, out_channels_fc0), nn.ReLU())

        M_conv1 = 9
        out_channels_conv1 = 32
        self.h_conv1 = CustomConv2d(out_channels_fc0, out_channels_conv1, M_conv1)

        M_conv2 = 9
        out_channels_conv2 = 64
        self.h_conv2 = CustomConv2d(out_channels_conv1, out_channels_conv2, M_conv2)

        M_conv3 = 9
        out_channels_conv3 = 128
        self.h_conv3 = CustomConv2d(out_channels_conv2, out_channels_conv3, M_conv3)

        out_channels_fc1 = 256  # 1024
        self.h_fc1 = nn.Sequential(CustomLinear(out_channels_conv3, out_channels_fc1), nn.ReLU())

        self.y_conv = CustomLinear(out_channels_fc1, self.num_classes)

    def forward(self, x, adj, pred_weight=None):
        _, n_pts, _ = x.shape

        x = self.h_fc0(x)  # (b, n_pts, c)
        x = self.h_conv1(x, adj, pred_weight.get('h_conv1'))
        x = self.h_conv2(x, adj, pred_weight.get('h_conv2'))
        x = self.h_conv3(x, adj, pred_weight.get('h_conv3'))
        x = self.h_fc1(x)
        y = self.y_conv(x)

        return y


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # input x: (b, 3, n_pts)
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (b, c, n_pts)

        # pool over all points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (b, c)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        iden = iden.to(x.device)
        x = x + iden  # identity matrix added to a pooled feature, produce a transformation matrix
        x = x.view(-1, 3, 3)  # (b, 3, 3)

        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        # input x: (b, k=64, n_pts)
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)
        iden = iden.to(x.device)
        x = x + iden  # identity matrix added to a pooled feature, produce a transformation matrix
        x = x.view(-1, self.k, self.k)  # (b, k=64, k=64)

        return x


class PointFeatNet(nn.Module):
    def __init__(self, global_feat=True, in_channels=3, point_transform=False, feature_transform=False):
        super(PointFeatNet, self).__init__()
        self.global_feat = global_feat
        self.in_channels = in_channels
        self.point_transform = point_transform
        self.feature_transform = feature_transform

        if self.point_transform:
            self.stn = STN3d()  # generate T-net to transform input

        self.conv1 = torch.nn.Conv1d(self.in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.fstn = STNkd(k=64)  # generate T-net to transform feature

    def forward(self, x, adj=None, weighted=None):
        # input x: (b, 3, n_pts)
        num_points = x.size()[2]

        # get T-net and transform input
        if self.point_transform:
            trans = self.stn(x)  # (b, 3, 3)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))  # (b, c, n_pts)

        # get T-net and transform feature
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        point_feat = x  # point-wise feature
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (b, ch, n_pts)

        if adj is None:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)  # global feature, (b, c)

            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
                x = torch.cat([x, point_feat], 1)  # concatenate global feature with point-wise feature
                return x, trans, trans_feat
        else:
            if weighted is None:
                x = x.transpose(1, 2)
                patches = get_patches(x, adj)  # (b, n_pts, K, ch)
                patches = torch.cat([patches, x.unsqueeze(2)], dim=2)
                x = torch.max(patches, 2)[0]
                x = x.transpose(1, 2)

                return x, trans, trans_feat
            else:
                x = x.transpose(1, 2)
                patches = get_patches(x, adj)  # (b, n_pts, K, ch)
                weighted = weighted.transpose(2, 3)  # (b, n_pts, c, K) -> (b, n_pts, K, c)
                x = torch.sum(patches * weighted, dim=2)
                x = x.transpose(1, 2)

                x = torch.cat([x, point_feat], 1)

                return x, trans, trans_feat


class OuterNet(nn.Module):
    def __init__(self, arch_file, in_channels=3, point_transform=False, feature_transform=False):
        super(OuterNet, self).__init__()
        self.arch_file = arch_file
        self.in_channels = in_channels
        self.point_transform = point_transform
        self.feature_transform = feature_transform

        self.feat = PointFeatNet(global_feat=False,
                                 in_channels=self.in_channels,
                                 point_transform=self.point_transform,
                                 feature_transform=self.feature_transform)

        self.feat_dim = 1024
        self.regressor = nn.ModuleDict()

        with open(self.arch_file, 'r') as f:
            self.arch = json.load(f)

        for layer_name in self.arch:
            layer_arch = self.arch.get(layer_name)

            for module_arch in layer_arch:
                module_name = module_arch.get('name')
                module_type = module_arch.get('type')
                in_ch = int(module_arch.get('in_ch'))
                out_ch = int(module_arch.get('out_ch'))

                if module_type == 'F':
                    module = nn.ModuleDict({module_name: nn.Sequential(OrderedDict([
                        ('weight', nn.Linear(self.feat_dim, out_ch * in_ch)),
                        ('bias', nn.Linear(self.feat_dim, out_ch))]))})

                    if not layer_name in self.regressor.keys():
                        self.regressor[layer_name] = nn.ModuleList()

                    self.regressor[layer_name].append(module)

    def forward(self, x, adj, return_feat=False):
        # input x: (b, n_pts, 3)
        batch_size, num_points, _ = x.shape
        x = x.transpose(1, 2)

        x, trans, trans_feat = self.feat(x, adj)

        x = x.transpose(1, 2)  # (b, ch, n_pts) -> (b, n_pts, ch)

        pred_weight = OrderedDict()
        for layer_name, layer in self.regressor.items():  # for each regressed layer
            for module_ind, module in enumerate(layer):  # for each module in a layer
                in_ch = int(self.arch.get(layer_name)[module_ind].get('in_ch'))
                out_ch = int(self.arch.get(layer_name)[module_ind].get('out_ch'))

                for module_name, module_wb in module.items():  # for weight and bias in a module
                    pred_w = module_wb.weight(x).view(batch_size, num_points, out_ch, in_ch)
                    pred_b = module_wb.bias(x).view(batch_size, num_points, out_ch)

                    if not layer_name in pred_weight.keys():
                        pred_weight[layer_name] = list()

                    pred_weight[layer_name].append({'weight': pred_w, 'bias': pred_b})

        if return_feat:
            return pred_weight, x
        else:
            return pred_weight