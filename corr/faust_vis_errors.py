import os
import glob
import h5py
import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from config import args

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

    for i in range(20):
        # with h5py.File(mat_off[i]) as f:
        # X = f['shape']['X']     # (1, 6890)
        # Y = f['shape']['Y']
        # Z = f['shape']['Z']
        #
        # X = X.value[0]  # (6890,)
        # Y = Y.value[0]
        # Z = Z.value[0]
        # triv = f['shape']['TRIV']   # edge index for n faces, (3, n)
        # triv = triv.value.T         # (n, 3)
        # triv = triv.astype('int32')
        #
        # xyz = np.asarray([X, Z, Y]).T   # (6890, 3)

        verts, faces = read_off(test_off[i])
        xyz = np.stack(verts, axis=0)
        triv = np.stack(faces, axis=0).astype(int)

        # triv = triv - 1

        # load geodesic distance
        distance = np.loadtxt(os.path.join(args.result_path, 'distances_{}.txt'.format(i)))[:6890]

        norm = mpl.colors.Normalize(vmin=distance.min(), vmax=distance.max())
        cmap = cm.hot(1 - norm(distance)) * 255

        mesh = trimesh.Trimesh(faces=triv, vertices=xyz, vertex_colors=cmap.astype('int32'), process=False)

        # mesh.visual.kind
        # mesh.show(viewer='gl')
        mesh.export(os.path.join(args.visual_path, 'errors', 'faust_error_{}.ply'.format(str(i + 80))))

        # ----------------------------------------------

        # load refined geodesic distance
        distance = np.loadtxt(os.path.join(args.result_path, 'distances_refine_{}.txt'.format(i)))[:6890]

        norm = mpl.colors.Normalize(vmin=distance.min(), vmax=distance.max())
        cmap = cm.hot(1 - norm(distance)) * 255

        mesh = trimesh.Trimesh(faces=triv, vertices=xyz, vertex_colors=cmap.astype('int32'), process=False)

        mesh.export(os.path.join(args.visual_path, 'errors', 'faust_error_refined_{}.ply'.format(str(i + 80))))


if __name__ == '__main__':
    # checkpoint
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # vis path
    if not os.path.exists(os.path.join(args.visual_path, 'errors')):
        os.makedirs(os.path.join(args.visual_path, 'errors'))

    run(args)