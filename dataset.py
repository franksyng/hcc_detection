import os
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

import utils
from utils import apply_clahe


class GOFandLOF(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path)
        label = np.stack((self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 2], self.annotations.iloc[idx, 3]), axis=-1)  # re-group labels
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class GeneAndTime(Dataset):
    def __init__(self, annotations, img_dir, grayscale: bool):
        self.annotations = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.grayscale = grayscale

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        gene_label contains [kras, bcat] only since all cases are P53.
        if want to add more gene labels:
        gene_label = np.stack(([self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 2], ..., self.annotations.iloc[idx, n]), axis=-1)
        """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = utils.load_img(img_path, self.grayscale, False)
        gene_label = np.stack(([self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 2]]), axis=-1) #
        time_label = np.stack(([self.annotations.iloc[idx, 4], self.annotations.iloc[idx, 5], self.annotations.iloc[idx, 6]]), axis=-1)  # re-group labels
        # print(gene_label)
        # print(time_label)
        return image, gene_label, time_label

