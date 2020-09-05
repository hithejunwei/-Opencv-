"""
此程序包括：
1.K-fold简单程序示例
2.SK-fold（分层交叉验证）简单示例
3.根据选择出的最优模型超参数选择最优模型参数
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
        sample_hist[i], _ = np.histogram(lbp, bins=max_bins,range=(0, max_bins),
                                    density=True)

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
# skfolds = StratifiedKFold(n_splits=5, shuffle=False,random_state=42)
# for train_index, test_index in skfolds.split(sample_hist, sample_label):
#     print(train_index, test_index)



# result=[]
# maxid=0
# max=0
# id=0
# kenellist=['linear', 'poly', 'rbf', 'sigmoid']
# mean_list=[]
#
# for kenel in kenellist :
#     for c in range(1,2001,100):
#         gama=0.1
#         while(gama<=1):
#             clf = SVC(kernel=kenel, C=c, gamma=gama)
#             score=cross_val_score(clf,sample_hist,sample_label,cv=5,
#                                   scoring="precision")
#             mean_list.append(score)
#             mean_score=np.mean(score)
#             result.append([kenel,c,gama,mean_score])
#             print(id,result[id])
#             if max<mean_score:
#                 max=mean_score
#                 maxid=id
#
#             gama=gama+0.1
#             id=id+1
#
# print("the best model:")
# print(result[maxid])
# print(mean_list[maxid])


"""
根据选择好的模型超参数，确定最优的模型参数（特征）
"""
skfolds = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
index_list = []
kf_result=[]
kf_id=0
kf_max=0
kf_max_id=0
for train_index, test_index in skfolds.split(sample_hist, sample_label):
    index_list.append(train_index)
    index_list.append(test_index)
    kf_train_data=[]
    kf_train_label=[]
    kf_test_data=[]
    kf_test_label=[]
    for i in range(len(train_index)):
        kf_train_data.append(sample_hist[train_index[i]])
        kf_train_label.append(sample_label[train_index[i]])

    for j in range(len(test_index)):
        kf_test_data.append(sample_hist[test_index[j]])
        kf_test_label.append(sample_label[test_index[j]])

    clf = SVC(kernel='linear', C=101, gamma=0.1)
    clf.fit(kf_train_data, kf_train_label)

    #保存模型
    # f = open('model/kf_svm'+str(kf_id)+'.pkl', "wb")
    # pickle.dump(clf, f)
    # f.close()


    test_predict = clf.predict(kf_test_data)
    precision=precision_score(kf_test_label, test_predict,
                                    average="macro")
    if kf_max < precision:
        kf_max=precision
        kf_max_id=kf_id

    kf_id+=1

    kf_result.append(precision)

print(kf_max_id)
print(kf_result)












