import numpy as np
import cv2 as cv
from skimage import feature as skft
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import pickle

def load_picture():

    test_index = 0
    test_data = np.zeros((4, 171, 171))
    test_label=[2,2,2,2]


    for i in range(4):
        image = cv.imread('F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\test/ex-1/test1/'+str(i)+'.tiff',0)
        test_data[test_index] = image
        test_index += 1

    return test_data,test_label


#提取图片特征
def feature_detect():

    test_hist = np.zeros( (4,256) )

    for i in np.arange(4):
        lbp = skft.local_binary_pattern(test_data[i],8,1)
        #统计图像的直方图
        max_bins = int(lbp.max()+1)
        #hist size:256
        test_hist[i], _ = np.histogram(lbp, bins=max_bins,range=(0, max_bins),density=True)


    return test_hist

test_data,test_label= load_picture()
test_hist = feature_detect()

f=open('F:\\360MoveData\\Users\\1\\Desktop\\zhinengzhizao\\cv\\cv-img\\model/clf_svm.pkl',"rb")
clf=pickle.load(f)


test_predict=clf.predict(test_hist)
print(test_predict)
print(classification_report(test_label, test_predict))
f.close()

# print(f1_score(test_label,test_predict,average="macro"))
# print(precision_score(test_label,test_predict,average="macro"))
# print(accuracy_score(test_label, test_predict))