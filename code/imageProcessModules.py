# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:44:09 2019

@author: Kazım
"""

import cv2
import numpy as np
import glob
import os
import resnet
import torch
import torchvision
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics


def imgShow(img):
    cv2.imshow("image",img)
    cv2.waitKey()
    cv2.destroyAllWindows()
        
        

def readImg(path):
    return cv2.imread(path,1)

def imgConvert2RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def imgPadder(img):
    
    shape = img.shape
    shape = (shape[0], shape[1])
    max_shape = max(shape)
    shape_half = (int(shape[0]/2),int(shape[1]/2))
    center = (int(max_shape/2),int(max_shape/2))
    img_final = np.zeros((max_shape,max_shape,3),dtype = np.uint8)
    img_final[center[0]-shape_half[0]:shape[0]+center[0]-shape_half[0],center[1]-shape_half[1]:shape[1]+center[1]-shape_half[1],:] = imgConvert2RGB(img)
    return img_final
 
def imgScale(img, dim):
    return cv2.resize(img,dim,cv2.INTER_LANCZOS4)

def imgAsFloat(img):
    return img/255.0

def imgNormalization(img,dim):
    img = imgPadder(img)
    img = imgAsFloat(img)
    img = imgScale(img,dim)
    
    red=0.485
    green=0.456
    blue=0.406
    
    red_weight = 0.229
    green_weight =  0.224
    blue_weight = 0.225
    
    img_final = img.copy()
    img_final[:,:,0] = (img[:,:,0] - red) / red_weight
    img_final[:,:,1] = (img[:,:,1] - green) / green_weight
    img_final[:,:,2] = (img[:,:,2] - blue) / blue_weight
    
    return img_final

def importAndProcess(dir_name,dim):
    
    dir_name = dir_name+"\\"
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(dir_name, '*\\*.JPEG'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = imgNormalization(readImg(img_path),dim)
            label = img_path.split("\\")
            label = label[1]
            imgs.append(img)
            labels.append(label)

            if len(imgs)%10 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass
    return imgs, labels

def featureNormalize(feature):
    return_list = []
    for i,feature_ in enumerate(feature):
        total = 0
        for j in range(np.size(feature_)):
            x = (feature_[0,j])
            total = total + x*x
        total = np.sqrt(total) + 1e-4
        return_value = feature_ / total
        return_list.append(return_value.reshape(return_value.shape[1]))
    return np.array(return_list)

def list2image(image_list):
    image_array = np.array(image_list,dtype=np.float32)
    image_array = np.transpose(image_array,[0,3,1,2])
    return image_array

def importResnet():
    return torchvision.models.resnet50(pretrained=True)

def saveObject(obj,file_name):
    fl = open(file_name, 'wb')
    pickle.dump( obj, fl )
    fl.close()
    
def loadObject(file_name):
    fl = open(file_name, 'rb')
    obj=pickle.load(fl)
    fl.close()
    return obj

def computeFeature(model,image):
    image = image.reshape(1,3,224,224)
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    feature_vector = model(image)
    
    return feature_vector.detach().numpy()
    
def labelLookUp(label):
    return np.unique(label)

def label2int(label,label_lookup):
    label_int_list = []
    for i,label_ in enumerate(label):
        label_int = np.where(label_ == label_lookup)[0][0]
        label_int_list.append(label_int)
    label_int = np.array(label_int_list)
    return label_int
    

