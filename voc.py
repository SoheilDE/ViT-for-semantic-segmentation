import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')

        with open(os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt'), 'r') as f:
            self.filenames = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_filename = os.path.join(self.img_dir, f'{self.filenames[index]}.jpg')
        mask_filename = os.path.join(self.mask_dir, f'{self.filenames[index]}.png')

        img = Image.open(img_filename).convert('RGB')
        mask = Image.open(mask_filename).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask