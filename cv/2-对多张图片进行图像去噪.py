"""
对高斯噪声和椒盐噪声进行滤波
"""

import cv2 as cv


for i in range(2):

    img= cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test\\ex-1/test2/"+str(i)+".tiff", 0)
    median = cv.medianBlur(img, 3)
    cv.imwrite("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test\\ex-1/test2-3/"+str(i)+".tiff",median)


for i in range(2,4):
    img = cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test\\ex-1/test2/" + str(i) + ".tiff", 0)
    gauss = cv.GaussianBlur(img, (5,5),0)
    cv.imwrite("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test\\ex-1/test2-3/" + str(i) + ".tiff", gauss)



