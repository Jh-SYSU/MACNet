import os
import pandas as pd
import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transforms import Pad2Square
from torchvision import transforms

from utils import make_csv, create_dir_if_not_exists
from dataset import TestDataset, TrainDataset
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='se_resnext50')
    parser.add_argument('--fold_index', nargs='+', type=int, default=[1])

    # args for path
    parser.add_argument('--test_image_dir', type=str, default='/data/yuedongyang/raojh/ODIR/ODIR-5K_Testing_Images/ODIR-5K_Testing_Images')
    parser.add_argument('--submission_sample', type=str, default='/data/yuedongyang/raojh/ODIR/submit_ODIR.csv')
    parser.add_argument('--submission_dir', type=str, default='../submissions/')

    # args for training
    parser.add_argument('--batch_size', type=int, default=32)

    # args for multiprocessing
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])

    config = parser.parse_args()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    config.submission_dir = os.path.join(config.submission_dir, now)
    create_dir_if_not_exists(config.submission_dir)

    print('submission_dir: %s' % config.submission_dir)

    config.num_classes = 61
    config.mean = [0.485, 0.456, 0.406]
    config.std = [0.229, 0.224, 0.225]

    config.gpu_ids = list(map(str, config.gpu_ids))

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(config.gpu_ids)
    config.gpu_ids = list(range(len(config.gpu_ids)))

    trans = transforms.Compose([Pad2Square(),
                            transforms.Resize(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize(config.mean, config.std)])

    dataset = TestDataset(data_dir=config.test_image_dir, transform=trans)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers)

    if config.model_name == 'se_resnext50':
        model = SE_ResNext50(config.num_classes)
    elif config.model_name == 'densenet201':
        model = DenseNet201(config.num_classes)
    else:
        raise RuntimeError("model_name must be one of [se_resnext50, densenet201]")

    model=nn.DataParallel(model,device_ids=config.gpu_ids)
    model = model.cuda()

    model_name = config.model_name

    for fold_index in config.fold_index:

        config.model_name = '%s[fold_%d]' % (model_name, fold_index)
        config.model_path = os.path.join(config.model_dir, config.model_name+'.pth')
        
        model.load_state_dict(torch.load(config.model_path))

        make_csv(model, dataset, loader, config)
