# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:29:17 2019

Function for creating feature points.

Input: Image normalised with imNorm
Output: Feature vector normalized to l2 distance.

@author: Artun
"""
from PIL import Image
import numpy as np
import imageio
import sys
import os
import torch
import resnet
import torchvision
import math

def extractFeat(image):

    # We append an augmented dimension to indicate batch_size, which is one
    image = np.reshape(image, [1, 224, 224, 3])
    # Model takes as input images of size [batch_size, 3, im_height, im_width]
    image = np.transpose(image2, [0, 3, 1, 2]) #dtype=float32
    # Convert the Numpy image2 to torch.FloatTensor (have multiple func. options)
    image = torch.FloatTensor(image2)
    # Extract features
    model = torchvision.models.resnet50(pretrained=True)
    feature_vector = model(image)
    # Feature_vector = feature_vector.cpu()
    # Convert the features of type torch.FloatTensor to a Numpy array
    feature_vector = feature_vector.detach().numpy()
    # Normalization
    sum = 0
    for i in range(0,np.size(feature_vector,1)):
        x = (feature_vector[0,i])
        sum = sum + x*x
        
    sum = math.sqrt(sum) + 1e-4
    feature_vector = feature_vector/sum

    return feature_vector
