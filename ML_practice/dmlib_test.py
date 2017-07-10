
# coding: utf-8

# In[1]:

import os
os.chdir(u'E:/量知/Ali/dmlib')
import pandas as pd
import numpy as np

train = pd.read_csv(u'E:/量知/ML/DATA/House_Price/train.csv') #house price predict
test = pd.read_csv(u'E:/量知/ML/DATA/House_Price/test.csv')


# ### 利用一些数据集测试dmlib中的各类函数

# ### sampling

# #### Weight_Sample 加权采样

# In[2]:

from sampling.WeightSample import WeightSample
ws = WeightSample(probCol ='SalePrice' ,weight=2 ,sampleRatio=0.5, replace=False).transform(train) #某一列加权，整体按采样比例随机采样
ws.head()


# #### Random Sample 随机采样

# In[3]:

from sampling.RandomSample import RandomSample
rs = RandomSample(sampleRatio=0.2, replace=False).transform(train)


# In[4]:

#---#
rs.head()


# In[12]:

len(rs)


# #### Stratified_Sample 分层采样

# In[13]:

from sampling.StratifiedSample import StratifiedSample
ss = StratifiedSample(strataColName='BsmtCond',sampleRatio=0.1 ).transform(train)
#--#
ss.head()


# ### setoperation

# #### Filter_and_Mapping 过滤与映射

# In[6]:

from setoperation.filterandmapping import FilterandMapping
f = FilterandMapping('SalePrice>10000')
r = f.transform(train)


# In[7]:

#--#
r.head()


# #### JOIN （左连接，右连接，内连接，全连接）

# In[2]:

s1 = pd.Series(np.array([1,2,3,4]))
s2 = pd.Series(np.array([2,4,5,6]))
s3 = pd.Series(np.array([1,3,5,7]))
df1 = pd.DataFrame({"A": s1 , "B": s2 , "C": s3 })
df2 = pd.DataFrame({"A":pd.Series(np.array([1,2,5,6])), "D":pd.Series(np.array([4,5,6,7])), "E":pd.Series(np.array([3,4,5,6])) })


# In[3]:

print "df1:\n",df1
print "df2:\n",df2


# In[4]:

from setoperation.Join import Join
j = Join(joinType='inner',leftColNames='A',rightColNames='A')
result = j.transform(df1, df2)


# In[5]:

#--#
result


# #### Union 行合并

# In[15]:

from setoperation.Union import Union
u = Union(leftSelectCol='A',rightSelectCol='A',Deduplicate=True)
result = u.transform(df1, df2)


# In[16]:

#--#
result


# #### ColumnsMerge 列合并

# In[17]:

from setoperation.ColumnsMerge import ColumnsMerge
m = ColumnsMerge(leftSelectCol='B',rightSelectCol='D',autoRenameCol=True)
cm = m.transform(df1, df2)


# In[18]:

cm


# ### preprocess

# #### AppendID  增加序号列

# In[19]:

from preprocess.AppendID import AppendID
a = AppendID()
appendID = a.transform(train)


# #### Normalize 归一化

# In[20]:

from preprocess.Normalize import Normalize
n = Normalize(featureColNames=['LotFrontage','MSSubClass'], keepOriginal=True)
normalized = n.transform(train)


# In[21]:

#--#
normalized.head()


# #### Standardize 标准化

# In[22]:

from preprocess.Standardize import Standardize
s = Standardize(featureColNames=['MSSubClass'],keepOriginal=True)
standardized = s.transform(train)


# In[23]:

#--#
standardized.head()


# #### TypeTransform

# In[3]:

from preprocess.TypeTransform import TypeTransform
tt = TypeTransform(double_selectCols=['MSSubClass'],double_default=0.0,keepOriginal=True)
tt2 = tt.transform(train)


# In[4]:

#--#
tt2.head()


# ### feature_engineer

# #### FeatureScale 特征尺度变换

# In[2]:

train.head()


# In[3]:

from feature_engineer.featurescale import FeatureScale
fs = FeatureScale(featureColNames=['LotArea'],categoryCols=None,labelCol=None,scaleMethod='log2')
fs2 = fs.transform(train)


# In[5]:

fs2.head()


# #### Feature_Soften特征异常平滑

# In[6]:

from feature_engineer.featuresoften import FeatureSoften
fs = FeatureSoften(featureColNames=['LotArea'],categoryCols=None,labelCol=None,softMethod=0,cl=3)
fs2 = fs.transform(train)


# #### One_Hot_Encoding 独热编码

# In[11]:

train['SaleType'].value_counts()


# In[2]:

from feature_engineer.onehotencoding import OneHotEncoding
oe = OneHotEncoding(binaryCols=['SaleType'])
oe2 = oe.transform(train)


# #### GBDT_Importance GBDT特征重要性评估

# In[2]:

iris = pd.read_csv(u"E:/量知/ML/DATA/IRIS.csv")
iris.head()


# In[4]:

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=3)
gbdt.fit(iris.drop('label',1), iris['label'])

from feature_engineer.GBDTimportance import GBDTImportance
gi = GBDTImportance(modelName=gbdt,labelColName='label',featureColNames=list(iris.drop('label',1).columns))
gbdt_importance = gi.transform(iris)


# #### LabelEncoding 

# ####  Regression_Importance回归模型特征重要性评估

# In[11]:

from feature_engineer.labelencoding import LabelEncoding
iris2 = LabelEncoding(featureColNames=['label']).transform(iris)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(iris2.drop('label',1), iris2['label'])

from feature_engineer.regressionimportance import RegressionImportance
ri = RegressionImportance(modelName=gbr,labelColName='label',featureColNames=list(iris.drop('label',1).columns))
r_importance = ri.transform(iris2)


# #### RandomForest_Importance 随机森林特征重要性

# In[14]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=10)
rf.fit(iris.drop('label',1), iris['label'])

from feature_engineer.randomForestImportance import RandomForestImportance
rfi = RandomForestImportance(modelName=rf, labelColName='label',featureColNames=list(iris.drop('label',1).columns))
rf_importance = rfi.transform(iris)


# #### FeatureDiscrete 特征离散

# In[4]:

from feature_engineer.featureDiscrete import FeatureDiscrete
fd = FeatureDiscrete(discreteCols=['sepal_length'], discreteMethod=1, maxBins=3,reserve=True)  #method=1: 等频离散
fd_iris = fd.transform(iris)


# In[5]:

fd_iris['sepal_length'].value_counts()


# In[3]:

from feature_engineer.featureDiscrete import FeatureDiscrete
fd = FeatureDiscrete(discreteCols=['sepal_length'], discreteMethod=0, maxBins=3,reserve=True)  #method=0: 等距离散
fd_iris = fd.transform(iris)


# In[6]:

from feature_engineer.featureDiscrete import FeatureDiscrete
fd = FeatureDiscrete(discreteCols=['sepal_length'],labelCol='label', discreteMethod=2, maxBins=3,reserve=True)  #method=2: 基于熵增益离散等距离散
fd_iris = fd.transform(iris)


# In[ ]:



