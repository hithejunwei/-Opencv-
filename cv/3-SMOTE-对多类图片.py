from imblearn.over_sampling import SMOTE
import cv2 as cv
import numpy as np
import os


def load_picture():

    train_data= np.zeros((18, 171*171))
    train_label= np.zeros((18))

    for i in range(9):
        img=cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-2/2/"+str(i)+".tiff", 0)
        img_new=img.ravel()
        train_label[i] = 2
        for j in range(29241):
            train_data[i,j] = img_new[j]

    for i in range(9,13):
        img = cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-2/3/" + str(i-9) + ".tiff", 0)
        img_new = img.ravel()
        train_label[i] = 3
        for j in range(29241):
            train_data[i, j] = img_new[j]

    for i in range(13,18):
        img = cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-2/4/" + str(i - 13) + ".tiff", 0)
        img_new = img.ravel()
        train_label[i] = 4
        for j in range(29241):
            train_data[i, j] = img_new[j]

    return train_data, train_label


train_data, train_label=load_picture()
model_smote= SMOTE(k_neighbors=3)
new_data, new_label=model_smote.fit_resample(train_data, train_label)

for i in range(18,27):
    img=new_data[i].reshape(171,171)
    cv.imwrite("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-2/sup/"+str(i)+".tiff",img)













