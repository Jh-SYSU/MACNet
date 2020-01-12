# -*- coding: utf-8 -*-
'''
Created on 2019.12.19

@author: Jiahua Rao
'''

import os
from PIL import Image

import torch
from torch.utils.data import Dataset



class TrainDataset(Dataset):
    def __init__(self, dataframe, train_data_dir, transform=None):
        super().__init__()
        self.df = dataframe[['左眼眼底图像','右眼眼底图像','N','D','G','C','A','H','M','O']].values
        self.train_data_dir = train_data_dir
        self.left_image_names = self.df[:, 0]   #左眼眼底图像
        self.right_image_names = self.df[:, 1]   #右眼眼底图像
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        labels = self.df[index][2:].astype('int64')
        left_image_names, right_image_names = self.left_image_names[index], self.right_image_names[index]
        left_image_path, right_image_path = os.path.join(self.train_data_dir, left_image_names), os.path.join(self.train_data_dir, right_image_names)

        with Image.open(left_image_path) as img:
            left_image = img.convert('RGB')
        with Image.open(right_image_path) as img:
            right_image = img.convert('RGB')

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        labels = torch.Tensor(labels)
        return left_image, right_image, labels


class TestDataset(Dataset):
    def __init__(self, dataframe, train_data_dir, transform=None):
        super().__init__()
        self.df = dataframe[['左眼眼底图像','右眼眼底图像']].values
        self.train_data_dir = train_data_dir
        self.left_image_names = self.df[:, 0]   #左眼眼底图像
        self.right_image_names = self.df[:, 1]   #右眼眼底图像
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        left_image_names, right_image_names = self.left_image_names[index], self.right_image_names[index]
        left_image_path, right_image_path = os.path.join(self.train_data_dir, left_image_names), os.path.join(self.train_data_dir, right_image_names)

        with Image.open(left_image_path) as img:
            left_image = img.convert('RGB')
        with Image.open(right_image_path) as img:
            right_image = img.convert('RGB')

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, labels
