import os
import glob
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pdb


class FAUST(Dataset):

    def __init__(self, path, max_neigh=10, mode='training'):
        self.path = path
        self.max_neigh = max_neigh
        self.mode = mode

        files = glob.glob(os.path.join(self.path, '*.npy'))     # 100 shapes in 10 poses
        files = np.sort(files)
        self.files = [np.load(sf, allow_pickle=True)[()] for sf in files]

        if self.mode == 'training':
            self.files = self.files[:80]
        else:
            self.files = self.files[80:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        loaded = self.files[idx]
        vertex, adj = loaded['vertex'], loaded['adj']   # (n_pts, 3), (n_pts, max_neigh) - valid vertex index in adj starting with 1

        if isinstance(adj, list):
            adj_padded = []
            for x in adj:
                if len(x) < self.max_neigh:
                    # adj_padded.append(list(x) + [-1] * (self.max_neigh - len(x)))
                    adj_padded.append(list(x) + [0] * (self.max_neigh - len(x)))
                else:
                    adj_padded.append(list(x)[:self.max_neigh])

            adj = np.stack(adj_padded, axis=0)

        return {'x': vertex, 'adj': adj}


if __name__ == '__main__':
    dataset_path = '../MPI-FAUST/training/xiang'

    max_neigh  = 6

    dataloader = DataLoader(FAUST(dataset_path, max_neigh, 'training'),
                            batch_size=4,
                            shuffle=True,
                            num_workers=0)

    for _, data in enumerate(dataloader):
        pdb.set_trace()
