import cv2
img = cv2.imread("你保存的人脸数据库.jpg")
print(img.shape)

#PCA降维
from sklearn.datasets import load_iris
from sklearn.decomposition import  PCA
iris = load_iris()
X = iris.data
y = iris.target
pcal = PCA(n_components=2)
pcal.fit(X)
pca2 = PCA(n_components=0.95)  #降维后保留的方差百分比为0.95
pca2.fit(X)
#指定使用MLE算法自动降维
pca3 = PCA(n_components='mle')
pca3.fit(X)
#查看模型训练后各项特征的方差
pcal.explained_variance_
#降维后特征的方差占比
pcal.explained_variance_ratio_
#查看指定，特征数的降维结果
x_pca = pcal.transform(X)
print(X.shape)
print(x_pca.shape)