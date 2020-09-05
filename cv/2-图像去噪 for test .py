"""
对高斯噪声和椒盐噪声进行滤波
"""

import cv2 as cv
from matplotlib import pyplot as plt

img= cv.imread("test/ex-1/test2-1/3.tiff", 0)
cv.imshow("original", img)
ax1=plt.subplot2grid((2,4), (0 ,0), colspan=2)
plt.hist(img.ravel(), 256, [0,256])

img_gauss = cv.imread('test/ex-1/test2/3.tiff',0)
cv.imshow("gauss-noise", img_gauss)
ax2=plt.subplot2grid((2,4), (0 ,2), colspan=2)
plt.hist(img_gauss.ravel(),256,[0,256])

# 高斯滤波
cv.namedWindow("gauss")

cv.createTrackbar("a", "gauss", 0, 100, lambda x: None)
cv.createTrackbar("b", "gauss", 0, 100, lambda x: None)

while(1):
    a=cv.getTrackbarPos("a","gauss")
    b=cv.getTrackbarPos("b","gauss")

    gauss = cv.GaussianBlur(img_gauss,(2*a+1, 2*a+1), 0)
    cv.imshow("gauss", gauss)
    if cv.waitKey(5)==ord("Q"):
        break



#均值滤波
# mean_blur=cv.blur(img_gauss,(5,1))
# cv.imshow("mean", mean_blur)


img_sp = cv.imread("test/ex-1/test2-1/0.tiff", 0)
cv.imshow("sp-noise", img_sp)
ax3=plt.subplot2grid((2,4), (1 ,0), colspan=2)
plt.hist(img_sp.ravel(),256,[0,256])
plt.show()

#中值滤波
median= cv.medianBlur(img_sp, 3)
cv.imshow("median", median)

cv.waitKey()

