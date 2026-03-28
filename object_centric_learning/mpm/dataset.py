import os
import json
import torch
import random
import numpy as np
import pickle as pk
from torch.utils.data import Dataset, DataLoader


class TetrominoesDataset(Dataset):
    def __init__(self, split, fp, seed):
        self.split = split
        data = np.load(fp, allow_pickle=True)['arr_0']
        # split the data into train/val/test set
        N = len(data)
        np.random.seed(seed)
        np.random.shuffle(data)
        print(f'random data split seed={seed}')

        if self.split == 'train':
            self.data = data[: int(0.8*N)]
        elif self.split == 'val':
            self.data = data[int(0.8*N) : int(0.9*N)]
        elif self.split == 'test':
            self.data = data[int(0.9*N) :]
        elif self.split == 'all':
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        try:
            image = d['image']
        except:
            image = ((d / 2.0) + 0.5) * 255
        sample = {
            'image': image,
        }
        return sample

if __name__ == '__main__':
    dataset = TetrominoesDataset(
        split='train',
        fp='tetrominoes.npz', 
    )
    print(len(dataset))
    loader = DataLoader(
        dataset, 
        shuffle=False, 
        num_workers=4,
        batch_size=128,
    )
    num_batches = 0
    for data in loader:
        num_batches += 1
        assert data['image'].shape[1:] == (35, 35, 3)
    print('Total batches:', num_batches)