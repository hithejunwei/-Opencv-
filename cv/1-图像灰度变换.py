"""
线性灰度变换+非线性灰度变换
"""

import cv2 as cv
import math

img=cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\img\\lena.jpg",0)
cv.imshow('input',img)


cv.namedWindow("output", cv.IMREAD_COLOR)

rows=img.shape[0]
cols=img.shape[1]

img_new=img.copy()

cv.createTrackbar("a", 'output', 0, 10, lambda x: None)
cv.createTrackbar("b", 'output', 0, 100, lambda x: None)

#线性变换

while(1):

    a = cv.getTrackbarPos("a", 'output')
    b = cv.getTrackbarPos("b", 'output')

    for i in range(rows):
        for j in range(cols):
            img_new[i,j]=a*img[i, j]+b


    cv.imshow('output',img_new)

    if cv.waitKey(5) ==ord("Q"):
        break


#非线性变换

#指数变换
# while(1):
#
#     a = cv.getTrackbarPos("a", 'output')
#
#     for i in range(rows):
#         for j in range(cols):
#             img_new[i,j]=a*math.log(img[i, j]+1)
#
#
#     cv.imshow('output',img_new)
#
#     if cv.waitKey(5) ==ord("Q"):
#         break

#gamma变换（指数）
# while(1):
#
#     a = cv.getTrackbarPos("a", 'output')
#     b = cv.getTrackbarPos("b", "output")
#
#     for i in range(rows):
#         for j in range(cols):
#             img_new[i,j]=a*pow(img[i, j],b)
#
#
#     cv.imshow('output',img_new)
#

#     if cv.waitKey(5) ==ord("Q"):
#         break
