
# coding: utf-8

# In[1]:

import os
os.chdir(u'E:/量知/Ali/dmlib')


# ### 利用同样的数据，测试不同算法的效果。
# ### 用到的算法有：Logistic Regression, Naive Bayes, Support Vector Machine, Gradient Boosting Decision Tree, KNN.

# ### import the libraries

# In[2]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# ### Reading the file

# In[3]:

data = pd.read_csv(u'E:/量知/ML/DATA/mushroom-classification/mushrooms.csv')


# In[4]:

data.head()


# In[5]:

data.describe()


# In[6]:

data.isnull().sum()


# In[7]:

data['cap-color'].unique()


# In[8]:

data['class'].unique()


# the mushrooms have two classes, poisonous or edible. 有毒或者可食用

# In[9]:

data.shape


# In[10]:

data1 = data.drop('class',1)


# ### 特征离散

# In[11]:

from feature_engineer.labelencoding import LabelEncoding


# In[12]:

le = LabelEncoding(list(data.columns))
data2 = le.transform(data)


# In[13]:

data2.head()


# In[14]:

data2.groupby('class').size()


# ### Plotting boxplot to see the distribution of the data

# In[15]:

# Create a figure instance
fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))

# Create an axes instance and the boxplot
bp1 = axes[0,0].boxplot(data2['stalk-color-above-ring'],patch_artist=True)

bp2 = axes[0,1].boxplot(data2['stalk-color-below-ring'],patch_artist=True)

bp3 = axes[1,0].boxplot(data2['stalk-surface-below-ring'],patch_artist=True)

bp4 = axes[1,1].boxplot(data2['stalk-surface-above-ring'],patch_artist=True)


# In[16]:

plt.show()


# In[17]:

data2['stalk-color-above-ring'].value_counts()


# 箱线图：http://blog.csdn.net/jia20003/article/details/6382347

# ### Separating features and label

# In[18]:

X = data2.iloc[:,1:23]
y = data2.iloc[:,0]


# In[19]:

X.describe()


# In[20]:

get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(16,14))
foo = sns.heatmap(data2.corr(),vmax=0.7,square=True,annot=True)


# ### Standardising the data

# In[21]:

#scale the data between -1 and 1
from preprocess.Standardize import Standardize
st = Standardize(featureColNames=list(X.columns))
st_X = st.transform(X)


# In[22]:

X.head()


# ### Principle Component Analysis

# PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。

# In[23]:

from feature_engineer.PCA import PrincipalComponentAnalysis
pca = PrincipalComponentAnalysis(featureColNames=list(X.columns),contriRate=1.0)
XX = pca.transform(st_X)


# In[24]:

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[25]:

covariance = pca.get_covariance()


# In[26]:

explained_variance_ratio=pca.explained_variance_ratio_  #返回 所保留的n个成分各自的方差百分比
explained_variance_ratio


# In[27]:

plt.figure(figsize=(6,4))
plt.bar(range(22), explained_variance_ratio, alpha=0.5, align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principle Components')
plt.legend(loc='best')
plt.tight_layout()


# the first 17 components retains more than 90% of the data

# ### take only first two principal components and visualise it using K-means clustering

# In[28]:

N = data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0], x[:,1])
plt.show()


# In[29]:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0: 'g', 1: 'y'}
label_color = [LABEL_COLOR_MAP[L] for L in X_clustered]
plt.scatter(x[:,0], x[:,1], c=label_color)
plt.show()


# thus using kmeans we are able to segregate 2 classes well using the first 2 components with maximum variance.

# # 模型构建

# ### Splitting the data into training and testing dataset

# In[30]:

from setoperation.Split import Train_test_split
tts =  Train_test_split(test_size=0.2,random_state=0)
X_train, X_test, y_train, y_test = tts.transform(data_frame=X, label=y)


# In[31]:

dataX = pd.concat([X_train,y_train],axis=1)


# In[32]:

X_test2 = X_test


# ### Logistic Regression

# In[33]:

from ml.classify import Logistic
clf1 = Logistic(featureColNames=list(X_train.columns), labelColName='class')
clf1.train(dataX)
result1 = clf1.transform(X_test)


# In[34]:

y_prob = X_test['predict_score']        # 预测为正类的概率


# In[35]:

y_prob.head()


# In[36]:

y_pred = np.where(y_prob > 0.5, 1, 0)        # 分类：阈值为0.5，大于0.5，分类为1；小于0.5，分类为0


# In[37]:

y_pred[0:5]


# In[38]:

clf1.score(X_test.drop(['label','predict_score'],1), y_pred)


# In[39]:

from evaluate.ConfusionMatrix import ConfusionMatrix
X_test['class'] = y_test
cm = ConfusionMatrix(labelColName='class',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(X_test)


# In[40]:

cmatrix


# In[41]:

class_names = ['Poisonous','edible']
def show_confusion_matrix(cmatrix, class_labels):
    plt.matshow(cmatrix,cmap=plt.cm.YlGn,alpha=0.7)
    ax = plt.gca()
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks(range(0,len(class_labels)))
    ax.set_xticklabels(class_labels,rotation=45)
    ax.set_ylabel('Actual Label', fontsize=16, rotation=90)
    ax.set_yticks(range(0,len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for row in range(len(cmatrix)):
        for col in range(len(cmatrix[row])):
            ax.text(col, row, cmatrix[row][col], va='center', ha='center', fontsize=16)
show_confusion_matrix(cmatrix, class_names)


# 从图中可以看出：有32个蘑菇是有毒的但预测为无毒，有41个蘑菇可食用但预测为有毒。

# In[42]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='class',scoreColName='predict_score')
ce.transform(X_test)


# In[43]:

print clf1


# ### Naive Bayes

# INPUT不能为负数

# In[44]:

from setoperation.Split import Split
s =  Split(fraction=0.2)
train,test = s.transform(data)


# In[45]:


from ml.classify import NaiveBayes
clf2 = NaiveBayes(featureColNames=list(X_train.columns), labelColName='class')
clf2.train(train)
result2 = clf2.transform(test.drop('class',1))


# In[46]:

print("Number of mislabeled points from %d points: %d"  %(len(result2), (test['class']!=result2['label']).sum()))


# In[47]:

from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
model_naive.fit(train.drop('class',1), train['class'])


# In[48]:

print("Number of mislabeled points from %d points: %d"  %(len(result2), (test['class']!=result2['label']).sum()))


# ### 交叉验证

# In[49]:

from sklearn.cross_validation import cross_val_score
Scores = cross_val_score( clf2, train.drop('class',1), train['class'], cv=5, scoring='accuracy')


# In[50]:

Scores


# In[51]:

Scores.mean()


# ### 混淆矩阵和ROC曲线

# In[52]:

from evaluate.ConfusionMatrix import ConfusionMatrix
result2['class'] = test['class']
cm = ConfusionMatrix(labelColName='class',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(result2)
cmatrix


# In[53]:

from sklearn import metrics
auc_roc = metrics.classification_report(result2['class'], result2['label'], target_names=['0','1'])
auc_roc


# In[54]:

auc_roc = metrics.roc_auc_score(result2['class'], result2['label'])
auc_roc


# In[55]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='class',scoreColName='predict_score')
ce.transform(result2)


# ## Support Vector Machine

# In[56]:

from ml.classify import SVM
clf3 = SVM(featureColNames=list(X_train.columns), labelColName='class')
clf3.train(train)
result3 = clf3.transform(test.drop('class',1))


# In[57]:

clf3


# ### 调参

# In[58]:

from sklearn.model_selection import GridSearchCV
param_grid = {'Cost':[1, 10, 100, 500, 1000], 'kernel':['linear','rbf'],
             'Cost':[1, 10, 100, 500, 1000], 'gramma':[1, 0.1, 0.01, 0.001, 0.0001],'kernel':['rbf']
             }    #C:对误分类的惩罚参数,默认为1
gs = GridSearchCV(estimator=clf3, param_grid=param_grid, scoring='accuracy', cv=5)
gs = gs.fit(train.drop('class',1), train['class'])
print gs.best_score_
print gs.best_params_


# In[59]:

clf3 = SVM(featureColNames=list(X_train.columns), labelColName='class',kernel='rbf',Cost=1,gramma=1)
clf3.train(train)
result3 = clf3.transform(test.drop('class',1))


# ### 混淆矩阵和ROC曲线

# In[60]:

from evaluate.ConfusionMatrix import ConfusionMatrix
result3['class'] = test['class']
cm = ConfusionMatrix(labelColName='class',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(result3)
cmatrix


# In[61]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='class',scoreColName='predict_score')
ce.transform(result3)


# ### Gradient Boosting Decision Tree

# In[62]:

from ml.classify import GBDT
clf4 = GBDT(featureColNames=list(X_train.columns), labelColName='class')
clf4.train(train)
result4 = clf4.transform(test.drop('class',1))


# In[63]:

clf4


# In[64]:

print("Number of mislabeled points from %d points: %d"  %(len(result4), (test['class']!=result4['label']).sum()))


# In[65]:

y_pred = result4['label']
y_test = test['class']
print "Accuarcy: %.4g" % metrics.accuracy_score(y_test, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, result4['predict_score'])


# ### 调参

# In[66]:

param_grid = {
    'treeNum': [100, 300, 500],
    'learnRate': [0.01, 0.1, 0.3],
    'maxDeep': [3, 5, 7],
    'maxLeafNum': [10,30,50],
    'max_features': ['sqrt','log2']  
}
gs = GridSearchCV(estimator = clf4, param_grid=param_grid, scoring='accuracy', cv=5 )
gs = gs.fit(train.drop('class',1), train['class'])
print gs.best_score_
print gs.best_params_


# In[71]:

clf4 = GBDT(featureColNames=list(X_train.columns), labelColName='class', treeNum=100, learnRate=0.01, maxDeep=3, maxLeafNum=10, max_features='sqrt')
clf4.train(train)
result4 = clf4.transform(test.drop('class',1))


# ### 混淆矩阵和ROC曲线

# In[68]:

from evaluate.ConfusionMatrix import ConfusionMatrix
result4['class'] = test['class']
cm = ConfusionMatrix(labelColName='class',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(result4)
cmatrix


# In[69]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='class',scoreColName='predict_score')
ce.transform(result4)


# ### KNN

# In[70]:

from ml.classify import KNN
clf5 = KNN(featureColNames=list(X_train.columns), labelColName='class')
clf5.train(train)
result5 = clf5.transform(test.drop('class',1))


# ### 混淆矩阵和ROC曲线

# In[72]:

from evaluate.ConfusionMatrix import ConfusionMatrix
result5['class'] = test['class']
cm = ConfusionMatrix(labelColName='class',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(result5)
cmatrix


# In[73]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='class',scoreColName='predict_score')
ce.transform(result5)


# In[ ]:



