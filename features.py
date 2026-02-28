import os
import cv2 as cv
import numpy as np
from pandas.core.common import random_state


def resi(image):  # 图像处理
    image = cv.resize(image, (64, 64))
    image = image.reshape(64 * 64)
    return image

data = []
labels = []

for path, dirs, files in os.walk('./需要识别的人的群体数据集'):
    label = path.split('\\')
    print(label)
    if len(label) > 1:
        print(label[-1])
        for file in files:
            file_path = os.path.join(path, file)
            img = cv.imread(file_path, 0)  # 读取灰度图像
            data.append(resi(img))
            labels.append(label[-1])

data_x = np.array(data)
data_y = np.array(labels)
# 将数据和标签转换为numpy数组
print(data_x.shape)
print(data_y.shape)
# #PCA降维
# from sklearn.decomposition import  PCA
# pca = PCA(n_components=60)
# pca.fit(data_x)
# pca.explained_variance_         #查看模型训练后各项特征的方差
# pca.explained_variance_ratio_      #降维后特征的方差占比
# #查看指定，特征数的降维结果
# x_pca = pca.transform(data_x)
# print(data_x.shape)
# print(x_pca.shape)
# #训练模型
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_x,
                                                 labels,random_state=42,test_size=0.2)   #特征：x_pca,   标签：labels
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train,y_train)
print(svc.score(x_test,y_test))
import joblib
joblib.dump(svc,'./face_svc.pkl')

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
