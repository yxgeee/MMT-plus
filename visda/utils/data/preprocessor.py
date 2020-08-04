from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
from PIL import ImageOps

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None,
                mutual=False, flip=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.flip = flip

        self.initialized = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.flip:
            img_flip = img.copy()
            img_flip = ImageOps.mirror(img_flip)

        if self.transform is not None:
            img = self.transform(img)
            if self.flip:
                img_flip = self.transform(img_flip)

        if not self.flip:
            return img, fname, pid, camid, index
        else:
            return img, img_flip, fname, pid, camid, index

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, pid, index
