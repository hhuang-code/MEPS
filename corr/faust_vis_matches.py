import os
import glob
import h5py
import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import args
from faust_data import FAUST
from faust_model import OuterNet, InnerNet

import pdb


def read_off(file_path):
    """Reads vertex coordinates and face indices from an OFF file."""
    with open(file_path) as file:
        # if 'OFF' != file.readline().strip():
        #     raise ('Not a valid OFF file')

        n_verts, n_faces = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]

    return verts, faces


def run(args):
    # mat_list = sorted(glob.glob('/home/mmvc/mmvc-ny-nas/Xiang_Li/LJ_LX/ICCV2019_TP-Net/EG16_tutorial/dataset/FAUST_registrations/meshes/diam=001/*.mat'))
    # mat_off = mat_list[80:]

    off_list = sorted(glob.glob('./MPI-FAUST/registrations/*.off'))
    test_off = off_list[80:]

    dataloader_test = DataLoader(FAUST(args.dataset_path, args.max_neigh, 'test'),
                                 batch_size=1,
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

    outer_net.eval()
    inner_net.eval()
    with torch.no_grad():

        ref_cmap = None

        for i, data in enumerate(dataloader_test):
            x, adj = data['x'].cuda(), data['adj'].cuda()  # NOTE: valid vertex index in adj starting with 1

            pred_weight = outer_net(x, adj)
            y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

            # with h5py.File(mat_off[i]) as f:

            # X = f['shape']['X']  # (1, 6890)
            # Y = f['shape']['Y']
            # Z = f['shape']['Z']
            # X = X.value[0]  # (6890,)
            # Y = Y.value[0]
            # Z = Z.value[0]

            # triv = f['shape']['TRIV']  # edge index for n faces, (3, n)
            # triv = triv.value.T  # (n, 3)
            # triv = triv.astype('int32')

            # xyz = np.asarray([X, Z, Y]).T  # (6890, 3)

            verts, faces = read_off(test_off[i])
            xyz = np.stack(verts, axis=0)
            triv = np.stack(faces, axis=0).astype(int)

            # triv = triv - 1

            norm = mpl.colors.Normalize(vmin=xyz.sum(axis=-1).min(), vmax=xyz.sum(axis=-1).max())
            cmap = cm.rainbow(norm(xyz.sum(axis=-1))) * 255   # (6890, 4)

            if i % 10 == 0:
                ref_cmap = cmap.copy()

            mesh = trimesh.Trimesh(faces=triv, vertices=xyz, vertex_colors=ref_cmap.astype('int32'), process=False)

            mesh.export(os.path.join(args.visual_path, 'matches', 'faust_gt_{}.ply'.format(str(i + 80))))

            # -------------------------------------------------------

            pred = y.max(dim=-1)[1].squeeze(0).cpu().numpy()

            pred_cmap = ref_cmap[pred]

            mesh = trimesh.Trimesh(faces=triv, vertices=xyz, vertex_colors=pred_cmap.astype('int32'), process=False)

            mesh.export(os.path.join(args.visual_path, 'matches', 'faust_pred_{}.ply'.format(str(i + 80))))

            # -------------------------------------------------------

            # load refined prediction
            filename = os.path.join(args.result_path, 'pred_refine_{}.txt'.format(i))
            with open(filename) as f:
                content = f.readlines()

            refined = np.stack([float(x.strip()) for x in content], axis=0).astype('int32')

            pred_cmap = ref_cmap[refined]

            mesh = trimesh.Trimesh(faces=triv, vertices=xyz, vertex_colors=pred_cmap.astype('int32'), process=False)

            mesh.export(os.path.join(args.visual_path, 'matches', 'faust_pred_refined_{}.ply'.format(str(i + 80))))


if __name__ == '__main__':
    # checkpoint
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # vis path
    if not os.path.exists(os.path.join(args.visual_path, 'matches')):
        os.makedirs(os.path.join(args.visual_path, 'matches'))

    run(args)