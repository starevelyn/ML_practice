
# coding: utf-8

# In[1]:

import os
os.chdir(u'E:/量知/Ali/dmlib')
import pandas as pd
import numpy as np

iris = pd.read_csv(u"E:/量知/ML/DATA/IRIS.csv")


# #### 查看数据集

# In[2]:

iris.head()


# In[3]:

iris.shape


# In[4]:

iris['label'].value_counts()


# #### 归一化

# In[5]:

from preprocess.Normalize import Normalize
n = Normalize(featureColNames=['sepal_length','sepal_width','petal_length','petal_width'], keepOriginal=False)
normalized = n.transform(iris)


# In[6]:

#--#
normalized.head()


# ### 模型构建

# ### 1.KMeans 聚类算法

# In[7]:

from ml.cluster import K_means
clf1 = K_means(featureColNames=list(normalized.drop('label',1)), centerCount=3,initCenterMethod='random',loop=100)
clf1.train(normalized)
predict = clf1.transform(normalized)


# #### 模型评估

# #### 预测的三种IRIS类别分布

# In[9]:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.FacetGrid(iris, hue="label", size=5)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend()


# #### 真实的三种IRIS类别分布

# In[10]:

sns.FacetGrid(iris, hue="predict", size=5)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend()


# In[ ]:

'''
from scipy.spatial.distance import cdist
d=cdist(X,Y,'euclidean')#假设X有M个元素，Y有N个元素，最终会生成M行N列的array，用来计算X、Y中每个相对元素之间的欧拉距离
numpy.min(d,axis=1) #如果d为m行n列，axis=0时会输出每一列的最小值，axis=1会输出每一行最小值
sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0] #求出平均畸变程度
'''


# #### K的选择：肘部法则

# 如果问题中没有指定k的值，可以通过肘部法则这一技术来估计聚类数量。肘部法则会把不同k值的成本函数值画出来。随着k值的增大，平均畸变程度会减小；每个类包含的样本数会减少，于是样本离其重心会更近。但是，随着k值继续增大，平均畸变程度的改善效果会不断减低。k值增大过程中，畸变程度的改善效果下降幅度最大的位置对应的k值就是肘部。

# In[21]:


import matplotlib
X = iris.drop(['label','predict'],1)
#计算K值从1到10对应的平均畸变程度：
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
K=range(1,10)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(
            cdist(X,kmeans.cluster_centers_,
                 'euclidean'),axis=1))/X.shape[0])

myfont = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel(u'平均畸变程度',fontproperties=myfont)
plt.title(u'用肘部法则来确定最佳的K值',fontproperties=myfont)


# #### 聚类效果的评价

# #### 轮廓系数（Silhouette Coefficient）:s =(b-a)/max(a, b)

# 轮廓系数是类的密集与分散程度的评价指标。它会随着类的规模增大而增大。彼此相距很远，本身很密集的类，其轮廓系数较大，彼此集中，本身很大的类，
# 其轮廓系数较小。轮廓系数是通过所有样本计算出来的，计算每个样本分数的均值，计算公式如下：
# s = (b - a)/ max(a,b)   -----a是每一个类中样本彼此距离的均值，b是一个类中样本与其最近的那个类的所有样本的距离的均值。

# In[26]:

from sklearn import metrics

plt.figure(figsize=(12,16))
plt.subplot(3,2,1)
X = iris.drop(['label','predict'],1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(u'样本',fontproperties=myfont)
x1 = iris["sepal_length"]
x2 = iris["sepal_width"]
plt.scatter(x1 ,x2 )
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
tests=[2,3,4,5,8]
subplot_counter=1
for t in tests:
    subplot_counter+=1
    plt.subplot(3,2,subplot_counter)
    kmeans_model=KMeans(n_clusters=t).fit(X)
#     print kmeans_model.labels_:每个点对应的标签值
    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],
             marker=markers[l],ls='None')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(u'K = %s, 轮廓系数 = %.03f' % 
                  (t, metrics.silhouette_score
                   (X, kmeans_model.labels_,metric='euclidean'))
                  ,fontproperties=myfont)


# ### 2. LogisticMultiClassification 

# In[28]:

from setoperation.Split import Split
iris_train, iris_test = Split(fraction=0.2).transform(iris)


# In[29]:

from ml.MultiClassify import LogisticMultiClassification
clf2 = LogisticMultiClassification(labelColName='label', featureColNames=['sepal_length','sepal_width','petal_length','petal_width'])
clf2.train(iris_train)
predict2 = clf2.transform(iris_test)


# In[34]:

#--#
predict2.head()


# In[64]:

result = predict2.reset_index(drop=True)


# In[65]:

#--#
result.head()


# In[69]:

n = 0
for i in range(len(predict2)):
    if(result.loc[i,'label'] == result.loc[i,'predict']):
        n +=1
accuracy = 1.0 * n/len(result)


# In[70]:

accuracy

