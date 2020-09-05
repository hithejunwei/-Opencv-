"""电水表识别侦测模型训练"""

import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from skimage import feature as skft
from sklearn.metrics import classification_report
import pickle
import os

##读取图片集，并将图片和标签一一对应
def load_picture():

    train_data= np.zeros((5, 3780*9))
    train_label= np.zeros((5))
    train_index= 0
    hog=cv.HOGDescriptor()

    for i in range(5):
        img=cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-3/1/"+str(i)+".tiff", 0)
        h, w = img.shape[:2] #第二个元素之前
        img_new = cv.resize(img, (128, int(128 * h / w)))
        hog_desc = hog.compute(img_new, winStride=(8, 8))
        fv = hog_desc.reshape(1, -1)
        train_data[train_index] =fv
        train_label[train_index] = 1
        train_index += 1

    return train_data, train_label

train_data, train_label=load_picture()

print(train_data)





