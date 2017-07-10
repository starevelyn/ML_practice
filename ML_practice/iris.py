
# coding: utf-8

# In[1]:

import os 
os.chdir(u'E:/量知/Ali/dmlib')


# # 问题定义

# ### Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

# In[2]:

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
 
from ml.classify import RandomForest
from pipeline import Pipeline
from feature_engineer.randomForestImportance import RandomForestImportance

if __name__ == '__main__':

    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=0.3)
    input = np.concatenate((X_train, y_train[:, np.newaxis]), axis=1)
    inputTable = pd.DataFrame(input, columns=['fea_a', 'fea_b', 'fea_c', 'fea_d', 'label'])

    f = RandomForest(labelColName='label', excludedColNames=[],treeNum=100,maxTreeDeep=None,randomColNum='log2',minNumObj=3)
    #f.train(inputTable)
    imp = RandomForestImportance(f, labelColName='label')
    p = Pipeline([f, imp])
    model = p.fit(inputTable)   #data = transformer.train(data)
    print(y_test.shape)
    print(model.transform(pd.DataFrame(X_test, columns=['fea_a', 'fea_b', 'fea_c', 'fea_d'])).ix[:, 'label'] == y_test)
    print(RandomForest.__mro__)


# In[ ]:



