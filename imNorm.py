# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:12:17 2019

@author: Artun
"""

import numpy as np
from PIL import Image

def imNorm( image ):

    #Open the image 
    image = image.convert("RGB")
    image_arr = np.array(image,dtype=int) # convert to a numpy array
    row_sz = np.size(image_arr,0)
    col_sz = np.size(image_arr,1)
    print("Row size is %s and Col size %s" % (row_sz, col_sz))
    
    
    #Padding to a square
    if row_sz > col_sz:
        diff = row_sz-col_sz
        zrs = np.zeros([row_sz,int(diff/2),3], dtype = int)
        image_pad = np.concatenate((zrs,image,zrs), axis = 1)
    
    else:
        diff = col_sz-row_sz
        zrs = np.zeros([int(diff/2),col_sz, 3], dtype = int)
        image_pad = np.concatenate((zrs,image,zrs), axis = 0)
    
    #Resize to a 224x224 window    
    image_pad_res_i=(Image.fromarray(np.uint8(image_pad),"RGB")).resize((224,224), resample=0)
    #image_pad_res_i.show()
    image_pad_res = np.array(image_pad_res_i,dtype=float)
    
    #Normalization of the image with respect to PyTorch reqs.
    image_pad_res = image_pad_res/255
    image_pad_res[:,:,0] = (image_pad_res[:,:,0]-0.485)/0.229
    image_pad_res[:,:,1] = (image_pad_res[:,:,1]-0.456)/0.224
    image_pad_res[:,:,2] = (image_pad_res[:,:,2]-0.406)/0.225
    
    #Output
    image_pad_res = np.float32(image_pad_res)

    return image_pad_res

    
    