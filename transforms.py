# -*- coding: utf-8 -*-
'''
Created on 2019.12.19

@author: Jiahua Rao
'''

import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms

class Pad2Square(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return transforms.CenterCrop(max(img.size))(img)
    
class CoarseDropout(object):
    def __init__(self, per_channel=False):
        self.per_channel = per_channel
    def __call__(self, img):
        drop = iaa.CoarseDropout(p=0.1,size_px=5,per_channel=self.per_channel)
        return Image.fromarray(drop.augment_image(np.array(img)))
    
class DiagRotate(object):
    def __init__(self, p=0.5):
        self.p = p
        pass
    def __call__(self, img):
        flag = np.random.rand() < self.p
        if flag:
            angle = np.arctan(img.size[0]/img.size[1])/np.pi*180
            return img.rotate(-angle)
        return img
