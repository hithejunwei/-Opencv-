import cv2 as cv
img=cv.imread("D:\\zhinengzhizao\\cv\\cv-img\\shouxie\\mohu.jpg",cv.IMREAD_GRAYSCALE)

cv.imshow('input',img)
cv.namedWindow('output',cv.WINDOW_NORMAL)
rows, cols=img.shape[:2]
maxnum=300
minnum=0

img_new=img.copy()

for i in range(rows):
    for j in range(cols):
        if img[i][j]>maxnum :
            maxnum=img[i][j]
        if img[i][j]<minnum :
            minnum=img[i][j]
for i in range(rows):
    for j in range(cols):
        img_new[i][j]=(255-0)/(maxnum-minnum)*(img_new[i][j]-minnum)
cv.imshow('output',img_new)
cv.waitKey(100000)