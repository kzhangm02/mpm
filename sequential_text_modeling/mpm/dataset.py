import os
import json
import torch
import random
import numpy as np
import pickle as pk
from torch.utils.data import Dataset, DataLoader


class APDataset(Dataset):
    def __init__(self, split, fp, seed):
        self.split = split
        data = np.load(fp)
        self.tokens = torch.Tensor(data['tokens']).long()
        self.masks = torch.Tensor(data['masks']).long()
        self.word_starts = torch.Tensor(data['word_starts']).long()
        N, self.M = self.tokens.shape
        idxs = np.arange(N)
        np.random.seed(seed)
        np.random.shuffle(idxs)
        print(f'random data split seed={seed}')

        if self.split == 'train':
            self.idxs = idxs[:1800]
        elif self.split == 'val':
            self.idxs = idxs[1800:2000]
        elif self.split == 'test':
            self.idxs = idxs[2000:]
        elif self.split == 'all':
            self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        sample = {
            'input_ids': self.tokens[i],
            'masks': self.masks[i],
            'word_starts': self.word_starts[i],
            'doc_ids': i,
        }
        return sample

if __name__ == '__main__':
    dataset = APDataset(
        split='train',
        fp='ap-bert.npz', 
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
        masks = data['masks']
        tokens = data['input_ids']
        assert (tokens[masks == 0] == 0).all()
        assert (masks[tokens == 0] == 0).all()
    print('Total batches:', num_batches)