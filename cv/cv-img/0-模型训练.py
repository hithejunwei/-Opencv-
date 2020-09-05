
import numpy as np
from sklearn.svm import SVC
from skimage import feature as skft
from sklearn.metrics import classification_report
import cv2 as cv
import pickle
import os


##读取图片集，并将图片和标签一一对应
def load_picture():

    train_data= np.zeros((100, 171, 171))
    test_data= np.zeros((80, 171, 171))
    train_label= np.zeros((100))
    test_label= np.zeros((80))
    train_index= 0
    test_index= 0

    for i in range(20):
        a=len(os.listdir("picture_train/"+str(i)))
        for j in range(a):
            img=cv.imread("picture_train/"+str(i)+"/"+str(j)+".tiff", 0)
            train_data[train_index] = img
            train_label[train_index] = i
            train_index += 1

        for k in range(4):
            img=cv.imread("picture_test/"+str(i)+"/"+str(k)+".tiff", 0)
            test_data[test_index] = img
            test_label[test_index] = i
            test_index += 1
    return train_data, test_data, train_label, test_label

def feature_detect():
    train_hist= np.zeros( (100,256) )
    test_hist= np.zeros((80, 256))
    for i in range(100):
        #使用LBP方法提取图片的纹理特征
        lbp=skft.local_binary_pattern(train_data[i], 8, 1)
        max_bins= int(lbp.max()+1)
        train_hist[i], _= np.histogram(lbp, bins=max_bins, range=(0,
                                                                  max_bins), density=True)

    for j in range(80):
        lbp=skft.local_binary_pattern(test_data[j], 8 ,1)
        max_bins=int(lbp.max()+1)
        test_hist[j], _= np.histogram(lbp, bins=max_bins, range=(0, max_bins),
                                      density=True)

    return train_hist, test_hist

train_data, test_data, train_label, test_label=load_picture()
train_hist, test_hist= feature_detect()

result=[]
maxid=0
max=0
id=0
kenellist=['linear', 'poly', 'rbf', 'sigmoid']
for kenel in kenellist :
    for c in range(1,2001,100):
        gama=0.1
        while(gama<=1):
            clf = SVC(kernel=kenel, C=c, gamma=gama)
            clf.fit(train_hist,train_label)
            linshi=clf.score(test_hist,test_label)
            result.append([kenel,c,gama,linshi])
            print(id,result[id])
            if max<linshi:
                max=linshi
                maxid=id

            gama=gama+0.1
            id=id+1

print("the best model:")
print(result[maxid])

clf = SVC(kernel=result[maxid][0], C=result[maxid][1], gamma=result[maxid][2])
clf.fit(train_hist,train_label)

test_predict=clf.predict(test_hist)


print(classification_report(test_label, test_predict))

#保存模型
f=open('model/clf_svm.pkl',"wb")
pickle.dump(clf,f)
f.close()









