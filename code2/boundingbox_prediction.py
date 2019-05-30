# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:09:08 2019

@author: Kazım
"""

from imageProcessModules import *
import cv2
import numpy as np
import glob
import os

svm_model = loadObject(file_name="svm_model.cs484")
svm_model = loadObject(file_name="model1.sav")

dir_name = "test"
dir_name = dir_name+"\\"
imgs = []
dim = (224,224)
all_img_paths = glob.glob(os.path.join(dir_name, '*\\*.JPEG'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    try:
        img = readImg(img_path)
        imgs.append(img)
        if len(imgs)%10 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
    except (IOError, OSError):
        print('missed', img_path)
        pass
    
model = "model.yml"
resnet_model = importResnet()
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
for img in imgs:
    
    
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    crop_label = 0
    crop_prob = 0
    main_boundingbox = None
    for b in boxes:
        x, y, w, h = b
        crop_img = img[y:y+h,x:x+w,:]
        cv2.imshow("a",crop_img)
        cv2.waitKey(0)
        crop_img = imgNormalization(crop_img,(224,224))
        crop_feat = computeFeature(resnet_model,crop_img)
        crop_prob_array = svm_model.predict(featureNormalize([crop_feat]))
        print(crop_prob_array)
#        if crop_prob_array.max() > crop_prob:
#            crop_prob = crop_prob_array.max()
#            crop_label = np.where(crop_prob_array.reshape(crop_prob_array.shape[1])==crop_prob)
#            main_boundingbox = b
#        cv2.rectangle(img, (main_boundingbox[0], main_boundingbox[1]), (main_boundingbox[0]+main_boundingbox[2], main_boundingbox[1]+main_boundingbox[3]), (0, 0, 0), 4, cv2.LINE_AA)
    
    cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
