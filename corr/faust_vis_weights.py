import os
import glob
import trimesh
import numpy as np
from scipy.linalg import eigh
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from config import args
from faust_data import FAUST
from faust_model import OuterNet, InnerNet

import pdb


def compute_vertex_normals(vertices, faces):
    """
    Compute the normal vector at each vertex of the mesh.
    """
    vertex_normals = np.zeros_like(vertices)
    for face in faces:
        v1, v2, v3 = vertices[face]
        e1 = v2 - v1
        e2 = v3 - v1
        face_normal = np.cross(e1, e2)
        vertex_normals[face] += face_normal
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    return vertex_normals

def compute_curvature_tensor(vertices, faces, vertex_normals):
    """
    Compute the curvature tensor at each vertex using quadric fitting.
    """
    num_vertices = len(vertices)
    curvature_tensors = np.zeros((num_vertices, 3, 3))

    for vertex_idx in range(num_vertices):
        vertex = vertices[vertex_idx]
        normal = vertex_normals[vertex_idx]

        # Find the 1-ring neighborhood of the vertex
        neighborhood = []
        for face in faces:
            if vertex_idx in face:
                neighborhood.extend([vertices[idx] for idx in face if idx != vertex_idx])

        neighborhood = np.array(neighborhood)
        num_neighbors = len(neighborhood)

        # Fit a quadric surface to the neighborhood
        A = np.zeros((num_neighbors, 6))
        A[:, 0] = (neighborhood[:, 0] - vertex[0]) ** 2
        A[:, 1] = (neighborhood[:, 1] - vertex[1]) ** 2
        A[:, 2] = (neighborhood[:, 2] - vertex[2]) ** 2
        A[:, 3] = 2 * (neighborhood[:, 0] - vertex[0]) * (neighborhood[:, 1] - vertex[1])
        A[:, 4] = 2 * (neighborhood[:, 0] - vertex[0]) * (neighborhood[:, 2] - vertex[2])
        A[:, 5] = 2 * (neighborhood[:, 1] - vertex[1]) * (neighborhood[:, 2] - vertex[2])

        b = np.sum(neighborhood * normal, axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        curvature_tensors[vertex_idx] = np.array([[coeffs[0], coeffs[3] / 2, coeffs[4] / 2],
                                                   [coeffs[3] / 2, coeffs[1], coeffs[5] / 2],
                                                   [coeffs[4] / 2, coeffs[5] / 2, coeffs[2]]])

    return curvature_tensors

def compute_principal_curvatures(curvature_tensors):
    """
    Compute the principal curvatures from the curvature tensors.
    """
    principal_curvatures = np.zeros((len(curvature_tensors), 2))
    for i, tensor in enumerate(curvature_tensors):
        eigenvalues, _ = eigh(tensor)
        principal_curvatures[i] = eigenvalues[:2]

    return principal_curvatures

def compute_mean_gaussian_curvatures(principal_curvatures):
    """
    Compute the mean and Gaussian curvatures from the principal curvatures.
    """
    mean_curvatures = np.mean(principal_curvatures, axis=1)
    gaussian_curvatures = np.prod(principal_curvatures, axis=1)

    return mean_curvatures, gaussian_curvatures


# def read_off(file_path):
#     """Reads vertex coordinates and face indices from an OFF file."""
#     with open(file_path) as file:
#         # if 'OFF' != file.readline().strip():
#         #     raise ('Not a valid OFF file')
#
#         n_verts, n_faces = tuple([int(s) for s in file.readline().strip().split(' ')])
#         verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
#         faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
#
#     return verts, faces


def read_off(file_path):
    """Reads a mesh from an OFF file, supporting files without the 'OFF' header."""
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if first_line == 'OFF':
            mesh = trimesh.load(file_path, file_type='off')
        else:
            # Read the rest of the file and prepend the expected first line
            content = file.read()
            content = 'OFF\n' + first_line + '\n' + content
            mesh = trimesh.load_mesh(trimesh.util.wrap_as_stream(content), file_type='off')

    return mesh


def run(args):
    ply_list = sorted(glob.glob('./MPI-FAUST/registrations/*.ply'))
    test_ply = ply_list[80:]

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

            pred_weight, feat = outer_net(x, adj, return_feat=True)
            y = inner_net(x, adj, pred_weight)  # (b, n_pts, n_classes)

            x = x.squeeze(0).cpu().numpy()  # (n_pts, ch)
            gt = np.arange(0, x.shape[0])

            # mesh = read_off(test_off[i])
            mesh = trimesh.load(test_ply[i], file_type='ply')

            # # for radius in [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0]:
            # for radius in [0.025, 0.05, 0.075]:
            #     gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)
            #
            #     gaussian_curvature_norm = mpl.colors.Normalize(vmin=gaussian_curvature.min(), vmax=gaussian_curvature.max())
            #     gaussian_curvature_cmap = cm.rainbow(gaussian_curvature_norm(gaussian_curvature)) * 255  # (6890, 4)
            #
            #     mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                            vertex_colors=gaussian_curvature_cmap.astype(int), process=False)
            #
            #     mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_gaussian_curvature_{:.3f}.obj'.format(str(i + 80), radius)))
            #
            # # for radius in [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.2]:
            # for radius in [0.025, 0.05, 0.075]:
            #     mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)
            #
            #     mean_curvature_norm = mpl.colors.Normalize(vmin=mean_curvature.min(), vmax=mean_curvature.max())
            #     mean_curvature_cmap = cm.rainbow(mean_curvature_norm(mean_curvature)) * 255  # (6890, 4)
            #
            #     mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                            vertex_colors=mean_curvature_cmap.astype(int), process=False)
            #
            #     mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_mean_curvature_{:.3f}.obj'.format(str(i + 80), radius)))

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
                    weight_cmap = cm.rainbow(weight_norm(weight_tsne)) * 255  # (6890, 4)

                    mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
                                           vertex_colors=weight_cmap.astype(int), process=False)

                    mesh.export(os.path.join(args.visual_path, 'curvature',
                                             'faust_{}_{}_{}_weight.obj'.format(str(i + 80), key, str(j + 1))))


                    bias_norm = mpl.colors.Normalize(vmin=bias_tsne.min(), vmax=bias_tsne.max())
                    bias_cmap = cm.rainbow(bias_norm(bias_tsne)) * 255  # (6890, 4)

                    mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
                                           vertex_colors=bias_cmap.astype(int), process=False)

                    mesh.export(os.path.join(args.visual_path, 'curvature',
                                             'faust_{}_{}_{}_bias.obj'.format(str(i + 80), key, str(j + 1))))


                    layer = np.concatenate([weight, bias], axis=-1)

                    layer_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(layer).squeeze(-1)  # (n_pts,)

                    layer_norm = mpl.colors.Normalize(vmin=layer_tsne.min(), vmax=layer_tsne.max())
                    layer_cmap = cm.rainbow(layer_norm(layer_tsne)) * 255  # (6890, 4)

                    mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
                                           vertex_colors=layer_cmap.astype(int), process=False)

                    mesh.export(os.path.join(args.visual_path, 'curvature',
                                             'faust_{}_{}_{}_layer.obj'.format(str(i + 80), key, str(j + 1))))

                    conv.append(layer)

                conv = np.concatenate(conv, axis=-1)

                conv_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(conv).squeeze(-1)  # (n_pts,)

                conv_norm = mpl.colors.Normalize(vmin=conv_tsne.min(), vmax=conv_tsne.max())
                conv_cmap = cm.rainbow(conv_norm(conv_tsne)) * 255  # (6890, 4)

                mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
                                       vertex_colors=conv_cmap.astype(int), process=False)

                mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_{}_conv.obj'.format(str(i + 80), key)))

                _all.append(conv)

            _all = np.concatenate(_all, axis=-1)

            all_tsne = TSNE(n_components=1, random_state=20200528).fit_transform(_all).squeeze(-1)  # (n_pts,)

            all_norm = mpl.colors.Normalize(vmin=all_tsne.min(), vmax=all_tsne.max())
            all_cmap = cm.rainbow(all_norm(all_tsne)) * 255  # (6890, 4)

            mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
                                   vertex_colors=all_cmap.astype(int), process=False)

            mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_all.obj'.format(str(i + 80))))

            ##################################################################################################

            # all_fro_norm = []
            # keys = list(pred_weight.keys())
            # for _, (key, value) in enumerate(zip(keys, pred_weight.values())):
            #     conv_fro_norm = []
            #     for j, param in enumerate(value):
            #         weight, bias = param['weight'], param['bias']
            #
            #         weight_fro_norm = (weight ** 2).sum((-1, -2)).sqrt().squeeze(0).detach().cpu().numpy()
            #
            #         bias_fro_norm = (bias ** 2).sum(-1).sqrt().squeeze(0).detach().cpu().numpy()
            #
            #         fro_norm = mpl.colors.Normalize(vmin=min(weight_fro_norm.min(), bias_fro_norm.min()),
            #                                         vmax=max(weight_fro_norm.max(), bias_fro_norm.max()))
            #         weight_cmap = cm.rainbow(fro_norm(weight_fro_norm)) * 255  # (6890, 4)
            #
            #         mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                                vertex_colors=weight_cmap.astype(int), process=False)
            #
            #         mesh.export(os.path.join(args.visual_path, 'curvature',
            #                                  'faust_{}_{}_{}_weight.obj'.format(str(i + 80), key, str(j + 1))))
            #
            #
            #         # bias_norm = mpl.colors.Normalize(vmin=bias_fro_norm.min(), vmax=bias_fro_norm.max())
            #         bias_cmap = cm.rainbow(fro_norm(bias_fro_norm)) * 255  # (6890, 4)
            #
            #         mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                                vertex_colors=bias_cmap.astype(int), process=False)
            #
            #         mesh.export(os.path.join(args.visual_path, 'curvature',
            #                                  'faust_{}_{}_{}_bias.obj'.format(str(i + 80), key, str(j + 1))))
            #
            #         layer_fro_norm = np.sqrt((weight_fro_norm ** 2 + bias_fro_norm ** 2))
            #
            #         layer_norm = mpl.colors.Normalize(vmin=layer_fro_norm.min(), vmax=layer_fro_norm.max())
            #         layer_cmap = cm.rainbow(layer_norm(layer_fro_norm)) * 255  # (6890, 4)
            #
            #         mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                                vertex_colors=layer_cmap.astype(int), process=False)
            #
            #         mesh.export(os.path.join(args.visual_path, 'curvature',
            #                                  'faust_{}_{}_{}_layer.obj'.format(str(i + 80), key, str(j + 1))))
            #
            #         conv_fro_norm.append(layer_fro_norm ** 2)
            #
            #     conv_fro_norm = np.sqrt(np.sum(np.stack(conv_fro_norm, axis=-1), axis=-1))
            #
            #     conv_norm = mpl.colors.Normalize(vmin=conv_fro_norm.min(), vmax=conv_fro_norm.max())
            #     conv_cmap = cm.rainbow(conv_norm(conv_fro_norm)) * 255  # (6890, 4)
            #
            #     mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                            vertex_colors=conv_cmap.astype(int), process=False)
            #
            #     mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_{}_conv.obj'.format(str(i + 80), key)))
            #
            #     all_fro_norm.append(conv_fro_norm ** 2)
            #
            # all_fro_norm = np.sqrt(np.sum(np.stack(all_fro_norm, axis=-1), axis=-1))
            #
            # all_norm = mpl.colors.Normalize(vmin=all_fro_norm.min(), vmax=all_fro_norm.max())
            # all_cmap = cm.rainbow(all_norm(all_fro_norm)) * 255  # (6890, 4)
            #
            # mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices,
            #                        vertex_colors=all_cmap.astype(int), process=False)
            #
            # mesh.export(os.path.join(args.visual_path, 'curvature', 'faust_{}_all.obj'.format(str(i + 80))))



if __name__ == '__main__':
    # checkpoint
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # vis path
    if not os.path.exists(os.path.join(args.visual_path, 'curvature')):
        os.makedirs(os.path.join(args.visual_path, 'curvature'))

    run(args)
