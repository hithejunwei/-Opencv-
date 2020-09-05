"""单张图片HOG特征提取"""

import cv2 as cv


image=cv.imread("F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-3/1/0.tiff",0)
hog = cv.HOGDescriptor()
h, w = image.shape[:2]
# img=cv.resize(image, (128,int(128*h/w)))
img=image
cv.imshow("11",img)

# 64x128x9 =34020
fv = hog.compute(img, winStride=(8, 8))
print(fv)

cv.waitKey()

