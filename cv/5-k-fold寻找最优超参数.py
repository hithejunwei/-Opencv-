
"""
此程序包括：
1.K-fold和SK-fold简单程序示例，展示两者的区别
2.利用交叉验证选择合适的超参数
"""


import numpy as np
from sklearn.svm import SVC
from  skimage import feature as skft
from sklearn.model_selection import cross_val_score
import cv2 as cv
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import pickle

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
        sample_hist[i], _ = np.histogram(lbp, bins=max_bins,range=(0, max_bins),density=True)

    return sample_hist


sample_data,sample_label= load_picture()
sample_hist = feature_detect()


"""
下面这段程序只是K-fold的示例代码，即可以简单的看一下，具体在使用的过程中我们发现有的时候不能对样本进行很好的划分，
举例说明：对于有2类的样本来说，下面的划分可能会让一类数据全部划分到了测试集，训练集中不包含此类数据。这样的划分肯定是错的。
"""
# kf=KFold(n_splits=2,shuffle=False,random_state=42)
# for train_index, test_index in kf.split(sample_hist):
#     print(train_index, test_index)

"""
下面的程序中k-fold是分层划分，相较于上面的程序有所改进。
举例说明：对于有2类的样本来说，参数k即对应n_splits依赖于样本中数量最少的样本的个数。即必须保证划分的每一个子集
都包含所有类型的样本。
"""
skfolds = StratifiedKFold(n_splits=5, shuffle=False,random_state=42)
for train_index, test_index in skfolds.split(sample_hist, sample_label):
    print(train_index, test_index)

"""
选择合适的超参数
"""
result=[]
maxid=0
max=0
id=0
kenellist=['linear', 'poly', 'rbf', 'sigmoid']
mean_list=[]

for kenel in kenellist :
    for c in range(1,2001,100):
        gama=0.1
        while(gama<=1):
            clf = SVC(kernel=kenel, C=c, gamma=gama)
            score=cross_val_score(clf,sample_hist,sample_label,cv=5,
                                  scoring="precision")
            mean_list.append(score)
            mean_score=np.mean(score)
            result.append([kenel,c,gama,mean_score])
            print(id,result[id])
            if max<mean_score:
                max=mean_score
                maxid=id
                maxclf=clf

            gama=gama+0.1
            id=id+1
clf.fit(sample_data,sample_label)
test_predict=clf.predict()
print("the best model:")
print(result[maxid])
print(mean_list[maxid])