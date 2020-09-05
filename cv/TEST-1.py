
import cv2 as cv
import math

img=cv.imread('img/lena.jpg',0)

cv.imshow('input',img)


rows=img.shape[0]
cols=img.shape[1]

img_new=img.copy()

a,b=10,20

for i in range(rows):
    for j in range(cols):
        img_new[i,j]=a*img[i, j]+b


cv.imshow('output',img_new)
cv.waitKey()