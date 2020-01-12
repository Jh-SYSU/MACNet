# -*- coding: utf-8 -*-
'''
Created on 2019.12.19

@author: Jiahua Rao
'''

import os
import argparse
import pandas as pd
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from cyclic_lr import CyclicLR
from focal_loss import FocalLoss
from utils import train_and_eval, create_dir_if_not_exists
from dataset import TrainDataset
from model import *
from transforms import Pad2Square, CoarseDropout

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='se_resnext50')
    parser.add_argument('--fold_index', nargs='+', type=int, default=[1])

    # args for path
    parser.add_argument('--train_image_dir', type=str, default='/data/yuedongyang/raojh/ODIR/ODIR-5K_Training_Images/ODIR-5K_Training_Dataset')
    parser.add_argument('--csv_dir', type=str, default='/data/yuedongyang/raojh/ODIR/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--log_dir', type=str, default='./log/')

    # args for training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--early_stopping', type=int, default=3000)
    parser.add_argument('--focal_loss', action='store_true', default=False)

    # args for multiprocessing
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[1])

    # args for cyclic learning rate
    parser.add_argument('--base_lr', nargs='+', type=float, default=[1e-5, 5e-5])
    parser.add_argument('--max_lr', nargs='+', type=float, default=[5e-5, 1e-3])
    parser.add_argument('--cycle_size', type=int, default=3)

    config = parser.parse_args()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    config.checkpoint_dir = os.path.join(config.checkpoint_dir, now)
    config.log_dir = os.path.join(config.log_dir, now)

    create_dir_if_not_exists(config.checkpoint_dir)
    create_dir_if_not_exists(config.log_dir)

    config.num_classes = 8
    config.mean = [0.485, 0.456, 0.406]
    config.std = [0.229, 0.224, 0.225]

    config.gpu_ids = list(map(str, config.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(config.gpu_ids)
    config.gpu_ids = list(range(len(config.gpu_ids)))

    trans_train = transforms.Compose([Pad2Square(),
                                      transforms.Resize(size=256),
                                      transforms.RandomCrop(224),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(config.mean, config.std)])


    trans_valid = transforms.Compose([Pad2Square(),
                                      transforms.Resize(size=256),
                                      transforms.RandomCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(config.mean, config.std)])



    for fold_index in config.fold_index:
        # output model path
        config.model_file = os.path.join(config.checkpoint_dir, "%s[fold_%d].pth" % (config.model_name, fold_index))

        # output log path
        config.log_file = os.path.join(config.log_dir, 'log[fold_%d].txt' % (fold_index))

        print('model_file: %s' % config.model_file)
        print('log_file: %s' %  config.log_file)

        train_csv_path = os.path.join(config.csv_dir, 'train_%d.csv' % fold_index)
        valid_csv_path = os.path.join(config.csv_dir, 'valid_%d.csv' % fold_index)

        df_train = pd.read_csv(train_csv_path)
        df_valid = pd.read_csv(valid_csv_path)

        config.step_size = config.cycle_size * int(df_train.shape[0]/config.batch_size)

        dataset_train = TrainDataset(dataframe=df_train, train_data_dir=config.train_image_dir, transform=trans_train)
        dataset_valid = TrainDataset(dataframe=df_valid, train_data_dir=config.train_image_dir, transform=trans_valid)

        loader_train = DataLoader(dataset=dataset_train, 
                                batch_size=config.batch_size, shuffle=True, 
                                num_workers=config.num_workers)
        loader_valid = DataLoader(dataset=dataset_valid, 
                                batch_size=config.batch_size, shuffle=False, 
                                num_workers=config.num_workers)

        if config.focal_loss:
            criterion = FocalLoss(num_class=config.num_classes, smooth=0.1)
        else:
            criterion = nn.BCEWithLogitsLoss()


        # define model and optimizer
        if config.model_name == 'se_resnext50':
            model = SE_ResNext50(config.num_classes)
            optimizer = optim.SGD([{'params': model.last_linear.parameters(), 'lr': config.base_lr[1]},
                            {'params': model.backbone.parameters()}], lr=config.base_lr[0], 
                            momentum=0.9)
            scheduler = CyclicLR(optimizer=optimizer,base_lr=config.base_lr, 
                                 max_lr=config.max_lr,step_size=config.step_size,mode='triangular2')
        elif config.model_name == 'se_resnext50_spatial':
            model = SE_ResNext50_Spatial(config.num_classes)
            optimizer = optim.SGD([{'params': model.last_linear.parameters(), 'lr': config.base_lr[1]},
                            {'params': model.backbone.parameters()}], lr=config.base_lr[0], 
                            momentum=0.9)
            scheduler = CyclicLR(optimizer=optimizer,base_lr=config.base_lr, 
                                 max_lr=config.max_lr,step_size=config.step_size,mode='triangular2')
        elif config.model_name == 'se_resnext50_spatial_image':
            model = SE_ResNext50_Spatial_Image(config.num_classes)
            optimizer = optim.SGD([{'params': model.last_linear.parameters(), 'lr': config.base_lr[1]},
                            {'params': model.backbone.parameters()}], lr=config.base_lr[0], 
                            momentum=0.9)
            scheduler = CyclicLR(optimizer=optimizer,base_lr=config.base_lr, 
                                 max_lr=config.max_lr,step_size=config.step_size,mode='triangular2')
        elif config.model_name == 'densenet201':
            model = DenseNet201(config.num_classes)
            optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                lr=config.base_lr[0], momentum=0.9)
            scheduler = CyclicLR(optimizer=optimizer,base_lr=config.base_lr[0], 
                                 max_lr=config.max_lr[0],step_size=config.step_size,mode='triangular2')
        elif config.model_name == 'resnet50':
            model = Resnet50(config.num_classes)
            optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                lr=config.base_lr[0], momentum=0.9)
            scheduler = CyclicLR(optimizer=optimizer,base_lr=config.base_lr[0], 
                                 max_lr=config.max_lr[0],step_size=config.step_size,mode='triangular2')
        else:
            raise RuntimeError('model name must be one of [resnet50, se_resnext50, densenet201]')

        model = nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.cuda()

        train_and_eval(model, scheduler, optimizer, criterion, loader_train, loader_valid, config)


