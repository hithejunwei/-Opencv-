
import numpy as np
from sklearn.svm import SVC
from skimage import feature as skft
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV as gs
import cv2 as cv
import pickle
import os


def load_picture():

    sample_index = 0
    sample_data = np.zeros((14, 171, 171))
    sample_label=[1,1,1,1,1,1,1,1,1,2,2,2,2,2]

    for i in range(14):
        image = cv.imread('F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-4/1/'+str(i)+'.tiff',0)
        sample_data[sample_index] = image
        sample_index += 1

    return sample_data,sample_label


#提取图片特征
def feature_detect():

    sample_hist = np.zeros( (14,256) )

    for i in np.arange(14):
        lbp = skft.local_binary_pattern(sample_data[i],8,1)
        #统计图像的直方图
        max_bins = int(lbp.max()+1)
        #hist size:256
        sample_hist[i], _ = np.histogram(lbp, bins=max_bins,range=(0, max_bins),
                                    density=True)

    return sample_hist


sample_data,sample_label= load_picture()
sample_hist = feature_detect()

param_dist = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'C':np.arange(1,2001,100),
        'gamma':np.arange(0.1,1.1,0.1)
        }
clf=SVC()
grid_search=gs(clf,param_dist,cv=3,scoring="precision")
grid_result=grid_search.fit(sample_hist,sample_label)
best_estimator = grid_result.best_estimator_
print(best_estimator)



#上述程序代替了下面的程序
# result=[]
# maxid=0
# max=0
# id=0
# kenellist=['linear', 'poly', 'rbf', 'sigmoid']
# for kenel in kenellist :
#     for c in range(1,2001,100):
#         gama=0.1
#         while(gama<=1):
#             clf = SVC(kernel=kenel, C=c, gamma=gama)
#             clf.fit(train_hist,train_label)
#             linshi=clf.score(test_hist,test_label)
#             result.append([kenel,c,gama,linshi])
#             print(id,result[id])
#             if max<linshi:
#                 max=linshi
#                 maxid=id
#
#             gama=gama+0.1
#             id=id+1
# print("the best model:")
# print(result[maxid])











