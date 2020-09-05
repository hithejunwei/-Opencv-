import numpy as np
from skimage import feature as skft
import cv2 as cv

img = cv.imread('F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\img/lbp.tiff',0)

lbp = skft.local_binary_pattern(img,8,1)

#下面的程序为对图片整个的进行LBP特征提取，不对图片进行子区域划分
img_lbp_feature, _ = np.histogram(lbp, bins=256,range=(0, 256),density=True)


#下面的程序为对图片进行子区域的划分，提取出所有子区域的LBP特征
# img_new = np.zeros((9, 57, 57))
# temp_img_lbp_feature=np.zeros((9,256))
# index = 0
# for row in np.arange(3):
#     for col in np.arange(3):
#         img_new[index] = lbp[57 * row:57 * (row + 1),
#                                57 * col:57 * (col + 1)]
#         temp_img_lbp_feature[index], _ = np.histogram(img_new[index], bins=256, range=(0, 256),density=True)
#
# img_lbp_feature=temp_img_lbp_feature.ravel()


#打印这张图片的LBP特征的直方图向量
print(img_lbp_feature)