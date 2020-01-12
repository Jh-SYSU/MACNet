# -*- coding: utf-8 -*-
'''
Created on 2019.12.19

@author: Jiahua Rao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from pretrainedmodels import models

class DenseNet201(nn.Module):
    def __init__(self, num_classes=40):
        super(DenseNet201, self).__init__()
        self.backbone = tv_models.densenet201(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1920 * 2, num_classes),
        )
        self._set_require_grad()

    def forward(self, left_x, right_x):
        left_features = self.backbone.features(left_x)
        right_features = self.backbone.features(right_x)

        features = torch.cat([left_features, right_features], 1)

        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=(7,7), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def _set_require_grad(self):
        for para in list(self.backbone.parameters()):
            para.requires_grad=False
        for para in list(self.backbone.features.denseblock3.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.features.transition3.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.features.denseblock4.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.features.norm5.parameters()):
            para.requires_grad=True
        for para in list(self.classifier.parameters()):
            para.requires_grad=True


class Resnet50(nn.Module):
    def __init__(self, num_classes=40):
        super(Resnet50, self).__init__()
        self.backbone = tv_models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, num_classes),
        )
        self._set_require_grad()

    def features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def logits(self, x):
        print(x.shape)
        x = self.backbone.avgpool(x)
        print(x.shape)
        assert 0
        x = x.view(x.size(0), -1)
        return x

    def forward(self, left_x, right_x):
        left_features = self.features(left_x)
        right_features = self.features(right_x)

        features = torch.cat([left_features, right_features], 1)

        out = self.logits(features)
        out = self.classifier(out)
        return out

    def _set_require_grad(self):
        for para in list(self.backbone.parameters()):
            para.requires_grad=False
        for para in list(self.backbone.conv1.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.bn1.parameters()):
            para.requires_grad=True  
        for para in list(self.backbone.maxpool.parameters()):
            para.requires_grad=True  
        for para in list(self.backbone.layer1.parameters()):
            para.requires_grad=True 
        for para in list(self.backbone.layer2.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.layer3.parameters()):
            para.requires_grad=True
        for para in list(self.backbone.layer4.parameters()):
            para.requires_grad=True
        for para in list(self.classifier.parameters()):
            para.requires_grad=True


class SE_ResNext50(nn.Module):
    def __init__(self, num_classes=40):
        super(SE_ResNext50, self).__init__()
        self.backbone = models.se_resnext50_32x4d(num_classes=1000, 
            pretrained='imagenet')
        in_features = self.backbone.last_linear.in_features
        self.last_linear = nn.Sequential(
            nn.Linear(in_features * 2, num_classes),
        )
        self._set_require_grad() 

    def features(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def logits(self, x):
        x = self.backbone.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, left_x, right_x):
        left_x = self.features(left_x)
        right_x = self.features(right_x)

        x = torch.cat([left_x, right_x], 1)

        features = self.logits(x)
        outputs = self.last_linear(features)
        return outputs

    def _set_require_grad(self):
        for para in list(self.backbone.parameters()):
            para.requires_grad=True
        for para in list(self.last_linear.parameters()):
            para.requires_grad=True


class SE_ResNext50_Spatial(nn.Module):
    def __init__(self, num_classes=40):
        super(SE_ResNext50_Spatial, self).__init__()
        self.backbone = models.se_resnext50_32x4d(num_classes=1000, 
            pretrained='imagenet')
        self.conv2d_1x1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )
        self.last_linear = nn.Sequential(
            nn.Linear(2048 * 2, num_classes),
        )
        self._set_require_grad() 

    def features(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def logits(self, x):
        x = self.backbone.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def spatial_level_module(self, left_x, right_x):
        F_left_v = self.conv2d_1x1(left_x)
        F_left_k = self.conv2d_1x1(left_x)
        F_left_q = self.conv2d_1x1(left_x)

        F_right_v = self.conv2d_1x1(right_x)
        F_right_k = self.conv2d_1x1(right_x)
        F_right_q = self.conv2d_1x1(right_x)

        R_right2left = self.sigmoid(torch.mul(F_left_k, F_right_q))
        R_left2right = self.sigmoid(torch.mul(F_right_k, F_left_q))

        F_left_update = torch.mul(F_left_v, R_right2left)
        F_right_update = torch.mul(F_right_v, R_left2right)

        spatial_left = torch.mul(left_x, F_left_update)
        spatial_right = torch.mul(right_x, F_right_update)
        # spatial_left = torch.cat([left_x, F_left_update], 1)
        # spatial_right = torch.cat([right_x, F_right_update], 1)
        return spatial_left, spatial_right

    def forward(self, left_x, right_x):
        left_x = self.features(left_x)
        right_x = self.features(right_x)

        spatial_left, spatial_right = self.spatial_level_module(left_x, right_x)
        x = torch.cat([spatial_left, spatial_right], 1)

        features = self.logits(x)
        outputs = self.last_linear(features)
        return outputs

    def _set_require_grad(self):
        for para in list(self.backbone.parameters()):
            para.requires_grad=True
        for para in list(self.last_linear.parameters()):
            para.requires_grad=True


class SE_ResNext50_Spatial_Image(nn.Module):
    def __init__(self, num_classes=40):
        super(SE_ResNext50_Spatial, self).__init__()
        self.backbone = models.se_resnext50_32x4d(num_classes=1000, 
            pretrained='imagenet')
        self.conv2d_1x1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )
        self.conv2d_1x1_4096 = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        self.last_linear = nn.Sequential(
            nn.Linear(2048 * 2, num_classes),
        )
        self._set_require_grad() 

    def features(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def logits(self, x):
        x = self.backbone.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def spatial_level_module(self, left_x, right_x):
        F_left_v = self.conv2d_1x1(left_x)
        F_left_k = self.conv2d_1x1(left_x)
        F_left_q = self.conv2d_1x1(left_x)

        F_right_v = self.conv2d_1x1(right_x)
        F_right_k = self.conv2d_1x1(right_x)
        F_right_q = self.conv2d_1x1(right_x)

        R_right2left = self.sigmoid(torch.mul(F_left_k, F_right_q))
        R_left2right = self.sigmoid(torch.mul(F_right_k, F_left_q))

        F_left_update = torch.mul(F_left_v, R_right2left)
        F_right_update = torch.mul(F_right_v, R_left2right)

        spatial_left = torch.mul(left_x, F_left_update)
        spatial_right = torch.mul(right_x, F_right_update)
        # spatial_left = torch.cat([left_x, F_left_update], 1)
        # spatial_right = torch.cat([right_x, F_right_update], 1)
        return spatial_left, spatial_right

    def image_level_module(self, left_x, right_x):
        alpha_left = self.conv2d_1x1(left_x)
        alpha_right = self.conv2d_1x1(right_x)

        S_left_image = torch.mul(left_x, alpha_left)
        S_right_image = torch.mul(right_x, alpha_right)

        S_fea = left_x + S_left_image + right_x + S_right_image

        S_left_cat = torch.cat([left_x, S_fea], 1)
        S_right_cat = torch.cat([right_x, S_fea], 1)

        beta_left = self.conv2d_1x1_4096(S_left_cat)
        beta_right = self.conv2d_1x1_4096(S_right_cat)

        features_left = torch.mul(left_x, torch.mul(alpha_left, beta_left)+1)
        features_right = torch.mul(right_x, torch.mul(alpha_right, beta_right)+1)

        return features_left+features_right

    def forward(self, left_x, right_x):
        left_x = self.features(left_x)
        right_x = self.features(right_x)

        spatial_left, spatial_right = self.spatial_level_module(left_x, right_x)
        # x = torch.cat([spatial_left, spatial_right], 1)
        x = self.image_level_module(spatial_left, spatial_right)
        features = self.logits(x)
        outputs = self.last_linear(features)
        return outputs

    def _set_require_grad(self):
        for para in list(self.backbone.parameters()):
            para.requires_grad=True
        for para in list(self.last_linear.parameters()):
            para.requires_grad=True
