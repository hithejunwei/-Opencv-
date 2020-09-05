"""
此程序包括：
1.LBP特征提取简单示例
2.LBP特征提取细化简单示例
"""


import numpy as np
from skimage import feature as skft
import cv2 as cv


def load_picture():

    test_index = 0
    test_data = np.zeros((5,171, 171))
    test_label=[1,1,1,1,1]

    for i in range(5):
        image = cv.imread('F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-3/1/'+str(i)+'.tiff',0)
        test_data[test_index] = image
        test_index += 1

    return test_data,test_label

#提取图片特征
def feature_detect():

    test_hist = np.zeros( (5,256) )

    for i in np.arange(5):
        #取半径为1的8个邻域点
        lbp = skft.local_binary_pattern(test_data[i],8,1)

        #若取的半径不为1，其领域点的个数大于8，则需要将取值范围归一化到[0, 255]区间内
        # for j in range(171):
        #     for k in range(171):
        #         lbp[j,k]=(lbp[j,k]-lbp.min())/(lbp.max()-lbp.min())*255

        # 统计图像的直方图
        max_bins = int(lbp.max()+1)
        #hist size:256
        test_hist[i], _ = np.histogram(lbp, bins=max_bins,range=(0, max_bins),
                                       density=True)

    return test_hist

# 若想要检测的粒度更细，我们可以将图像的LBP特征图像进行划分，计算每一个划分区域的LBP图像对应的直方图，再将得到的所有的直方图向量合并
# def feature_detect():
#
#     test_hist= np.zeros((5,256*9))
#
#     for i in range(5):
#
#         lbp = skft.local_binary_pattern(test_data[i], 8, 1)
#
#         img_new = np.zeros((9, 57, 57))
#         temp_test_hist=np.zeros((9,256))
#         index = 0
#
#         for row in np.arange(3):
#             for col in np.arange(3):
#                 img_new[index] = lbp[57 * row:57 * (row + 1),
#                                        57 * col:57 * (col + 1)]
#                 temp_test_hist[index], _ = np.histogram(img_new[index],
#                                                         bins=256,
#                                                range=(0, 256),
#                                                density=True)
#
#                 index += 1
#
#
#                 # arr = 'test/ex-3/2/' + str(i) + str(index) + '.tiff'
#                 # cv.imwrite(arr, lbp[57 * row:57 * (row + 1),
#                 #                 57 * col:57 * (col + 1)])
#
#         test_hist[i]=temp_test_hist.ravel()
#
#     return test_hist


test_data,test_label= load_picture()
test_hist = feature_detect()

print(test_hist)





