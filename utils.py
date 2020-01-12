# -*- coding: utf-8 -*-
'''
Created on 2019.12.19

@author: Jiahua Rao
'''

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

def json_to_df(json_path):
    with open(json_path) as f:
        json_content = json.load(f)
        df = pd.DataFrame(json_content)
    return df

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_log(log_str, fp):
    fp.write(log_str)
    fp.write('\n')
    print(log_str)

def make_header(config, fp):
    for key, value in vars(config).items():
        log_str = str(key) + ':' + str(value) + '\t'
        fp.write(log_str)
    fp.write('\n')

def train_and_eval(model, scheduler, optimizer, criterion, loader_train, loader_valid, config):
    best_val_acc = 0.0
    best_epoch = 0
    epoch_since_best = 0
    fp = open(config.log_file, 'w')
    make_header(config, fp)
    for epoch in range(config.num_epochs):
        model.train()
        train_total_samples = 0
        train_total_labels = 0
        train_acc = 0
        train_loss = 0
        for i, data in enumerate(tqdm(loader_train)):
            scheduler.batch_step()
            left_inputs, right_inputs, labels = data
            train_total_samples += labels.size()[0]
            train_total_labels += labels.size()[0] * labels.size()[1]
            left_inputs = Variable(left_inputs.cuda(device=config.gpu_ids[0]))
            right_inputs = Variable(right_inputs.cuda(device=config.gpu_ids[0]))
            labels = Variable(labels.cuda(device=config.gpu_ids[0]))

            optimizer.zero_grad()
            outputs = model(left_inputs, right_inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_pred = torch.round(torch.sigmoid(outputs))
            train_acc += torch.sum(train_pred == labels)
            train_loss += loss.item() * labels.size()[0]
                
        model.eval()
        valid_total_samples = 0
        valid_total_labels = 0
        valid_acc = 0
        val_loss = 0
        for _, data in enumerate(tqdm(loader_valid)):
            left_inputs, right_inputs, labels = data
            valid_total_samples += labels.size()[0]
            valid_total_labels += labels.size()[0] * labels.size()[1]
            left_inputs = Variable(left_inputs.cuda(device=config.gpu_ids[0]))
            right_inputs = Variable(right_inputs.cuda(device=config.gpu_ids[0]))
            labels = Variable(labels.cuda(device=config.gpu_ids[0]))

            optimizer.zero_grad()
            outputs = model(left_inputs, right_inputs)

            loss = criterion(outputs, labels)
            
            prediction = np.append(prediction, outputs.detach().cpu().numpy(), axis=0)
            valid_pred = torch.round(torch.sigmoid(outputs))
            valid_acc += torch.sum(valid_pred == labels)
            val_loss += loss.item() * labels.size()[0]

        train_acc = train_acc.cpu().numpy() / train_total_labels
        valid_acc = valid_acc.cpu().numpy() / valid_total_labels
        train_loss = train_loss / train_total_samples
        val_loss = val_loss / valid_total_samples
        
        log_str = '[Epoch %d] train loss %.6f train acc %.6f  valid loss %.6f valid acc %.6f' % (
            epoch, train_loss, train_acc, val_loss, valid_acc)
        make_log(log_str, fp)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch
            epoch_since_best = 0
            print('save %s' % config.model_file)
            torch.save(model.state_dict(), config.model_file)
        else:
            epoch_since_best += 1
            
        if epoch_since_best >= config.early_stopping:
            break
    
    make_log('Finished Training', fp)
    make_log('best_epoch: %d, best_val_acc %.6f' % (best_epoch, best_val_acc), fp)

    fp.close()

def predict_prob(model, loader, config):
    prediction = np.empty([0, config.num_classes])
    model.eval()
    for _, data in enumerate(tqdm(loader)):
        left_inputs, right_inputs = data
        left_inputs = Variable(left_inputs.cuda(device=config.gpu_ids[0]))
        right_inputs = Variable(right_inputs.cuda(device=config.gpu_ids[0]))
        outputs = model(left_inputs, right_inputs)
        prediction = np.append(prediction, outputs.detach().cpu().numpy(), axis=0)
    prob = prediction.reshape(-1, config.num_classes)
    prob = torch.sigmoid(torch.FloatTensor(prob)).numpy()
    return prob

def make_csv(model, dataset, loader, config):
    prob = predict_prob(model, loader, config)
    submit = pd.read_csv(config.submission_sample)

    df = pd.DataFrame(prob, columns=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    df['ID'] = submit.ID
    df.to_csv(os.path.join(config.submission_dir, config.model_name + '.csv'), index=False)