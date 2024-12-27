import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pdb


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]  # (n_shapes, n_pts, 3)
    label = f['label'][:]  # (n_shapes, 1)
    seg = f['pid'][:]  # (n_shapes, n_pts)

    return (data, label, seg)


def load_pts_seg_files(pts_file, seg_file, catid, cpid2oid):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)

    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([cpid2oid[catid + '_' + str(x)] for x in part_ids])

    return pts, seg


def pc_normalize(pc):
    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc


# def pc_augment_to_point_num(pts, pn):
#     assert (pts.shape[0] <= pn)
#     cur_len = pts.shape[0]
#     res = np.array(pts)
#     while cur_len < pn:
#         res = np.concatenate((res, pts))
#         cur_len += pts.shape[0]
#
#     return res[:pn, :]


# Ref: https://github.com/guochengqian/openpoints/blob/baeca5e319aa2e756d179e494469eb7f5ffd29f0/dataset/shapenetpart/shapenetpart.py#L52
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

    return translated_pointcloud


class ShapeNet(Dataset):
    def __init__(self, root, num_points=2048, mode='train'):
        self.root = root
        self.num_points = num_points
        self.mode = mode

        if self.mode in ['train']:
            train_file_list = os.path.join(self.root, 'train_hdf5_file_list.txt')
            val_file_list = os.path.join(self.root, 'val_hdf5_file_list.txt')
            self.file_list = (
                    [line.rstrip() for line in open(train_file_list)] + [line.rstrip() for line in open(val_file_list)])

            # if self.mode == 'train':
            #     train_file_list = os.path.join(root, 'train_hdf5_file_list.txt')
            #     self.file_list = [line.rstrip() for line in open(train_file_list)]
            # elif self.mode == 'val':
            #     val_file_list = os.path.join(root, 'val_hdf5_file_list.txt')
            #     self.file_list = [line.rstrip() for line in open(val_file_list)]

            total_data, total_labels, total_seg = [], [], []
            for file in self.file_list:
                cur_filename = os.path.join(self.root, file)
                cur_data, cur_labels, cur_seg = load_h5_data_label_seg(cur_filename)
                total_data.append(cur_data)
                total_labels.append(np.squeeze(cur_labels))
                total_seg.append(cur_seg)

            self.data = np.concatenate(total_data, axis=0)  # (n_shapes, n_pts, 3)
            self.labels = np.concatenate(total_labels, axis=0)  # (n_shapes,)
            self.seg = np.concatenate(total_seg, axis=0)  # (n_shapes, n_pts)
        else:
            if self.mode == 'test':
                # self.hdf5_data_dir = os.path.join(self.root, 'hdf5_data')
                # self.ply_data_dir = os.path.join(self.root, 'PartAnnotation')
                #
                # # part in each category to index, len(cpid2oid) = 50
                # self.cpid2oid = json.load(open(os.path.join(self.hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))
                #
                # # list of all test files
                # test_file_list = os.path.join(self.root, 'testing_ply_file_list.txt')
                # ffiles = open(test_file_list, 'r')
                # lines = [line.rstrip() for line in ffiles.readlines()]
                # self.pts_files = [line.split()[0] for line in lines]  # point files
                # self.seg_files = [line.split()[1] for line in lines]  # segment files
                # self.labels = [line.split()[2] for line in lines]  # labels for all shapes
                # ffiles.close()
                #
                # all_obj_cat_file = os.path.join(self.hdf5_data_dir, 'all_object_categories.txt')
                # fin = open(all_obj_cat_file, 'r')
                # lines = [line.rstrip() for line in fin.readlines()]
                # self.objcats = [line.split()[1] for line in lines]
                # # objnames = [line.split()[0] for line in lines]
                # # category to index, e.g., '02691156': 0
                # self.on2oid = {self.objcats[i]: i for i in range(len(self.objcats))}
                # fin.close()

                test_file_list = os.path.join(self.root, 'test_hdf5_file_list.txt')
                self.file_list = ([line.rstrip() for line in open(test_file_list)])

                total_data, total_labels, total_seg = [], [], []
                for file in self.file_list:
                    cur_filename = os.path.join(self.root, file)
                    cur_data, cur_labels, cur_seg = load_h5_data_label_seg(cur_filename)
                    total_data.append(cur_data)
                    total_labels.append(np.squeeze(cur_labels))
                    total_seg.append(cur_seg)

                self.data = np.concatenate(total_data, axis=0)  # (n_shapes, n_pts, 3)
                self.labels = np.concatenate(total_labels, axis=0)  # (n_shapes,)
                self.seg = np.concatenate(total_seg, axis=0)  # (n_shapes, n_pts)

    def __getitem__(self, index):
        # if self.mode in ['train', 'val']:
        #     pointclouds = self.data[index]  # (n_pts, 3)
        #     category = self.labels[index]  # scalar
        #     seg = self.seg[index]  # (n_pts,)
        #
        #     return {'id': index, 'x': pointclouds, 'category': category, 'label': seg}
        # else:
        #     if self.mode == 'test':
        #         cur_gt_category = self.on2oid[self.labels[index]]
        #
        #         pts_file_to_load = os.path.join(self.ply_data_dir, self.pts_files[index])
        #         seg_file_to_load = os.path.join(self.ply_data_dir, self.seg_files[index])
        #
        #         # pts: (n_pts, 3); seg: (n_pts,)
        #         pts, seg = load_pts_seg_files(pts_file_to_load, seg_file_to_load,
        #                                       self.objcats[cur_gt_category], self.cpid2oid)
        #
        #         aug_pts = pc_augment_to_point_num(pc_normalize(pts), self.num_points)
        #
        #         return {'id': index, 'x': aug_pts, 'category': cur_gt_category, 'label': seg, 'pts': pts}

        pointclouds = self.data[index]  # (n_pts, 3)
        category = self.labels[index]  # scalar
        seg = self.seg[index]  # (n_pts,)

        pointclouds = pc_normalize(pointclouds)

        if self.mode == 'train':
            pointclouds = translate_pointcloud(pointclouds)

        return {'id': index, 'x': pointclouds, 'category': category, 'label': seg}

    def __len__(self):
        # if self.mode in ['train']:
        #     return len(self.data)
        # else:
        #     if self.mode == 'test':
        #         return len(self.pts_files)
        return len(self.data)


if __name__ == '__main__':
    dataset_path = './shapenet_part_seg_hdf5_data'
    num_points = 2048
    batch_size = 8

    dataloader = DataLoader(ShapeNet(dataset_path, num_points, 'train'),
                                batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=0)

    for data in dataloader:
        pdb.set_trace()
