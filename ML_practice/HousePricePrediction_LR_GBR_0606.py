
# coding: utf-8

# In[1]:

import os
os.chdir(u'E:/量知/Ali/dmlib')


# In[2]:

import pandas as pd
import numpy as np


# ### 数据集--房屋价格预测数据集

# In[3]:

train = pd.read_csv(u'E:/量知/ML/DATA/House_Price/train.csv')


# In[4]:

train.shape


# In[5]:

train.head()


# # Fill missing values

# In[6]:

isnull = train.isnull().sum()
#print isnull


# #### 存在缺失值的列

# In[7]:

print isnull[isnull!=0]


# #### 将上述特征列的缺失值个数和占总体比例 按从大到小排列

# In[8]:

total = isnull.sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])


# In[9]:

#--#
missing_data.head(20)


# 缺失值太多的特征信息量太少，舍去不用。这样的特征有Alley，FireplaceQu，PoolQC，Fence，MiscFeature,LotFrontage

# In[10]:

train1 = train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','LotFrontage'],axis=1)


# In[54]:

train1_copy = train1.copy()


# #### 画出相关系数的热力图

# In[11]:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=[20,18])
sns.heatmap(train1.corr(), vmax=0.6, square=True, annot=True)


# 查看GarageCond的分布，填充缺失值

# In[12]:

#--#
train1['GarageCond'].value_counts()


# In[13]:

from preprocess.FillMissingValues import FillMissingValues
f = FillMissingValues('GarageCond', alter_method='ud', user_defined='TA')
train1 = f.transform(train1)


# 查看GarageType的分布，填充缺失值

# In[14]:

#--#
train['GarageType'].value_counts()


# In[15]:

f = FillMissingValues('GarageType', alter_method='ud', user_defined='Attchd')
train1 = f.transform(train1)


# 查看GarageYrBlt的分布，填充缺失值

# In[16]:

#--#
#train['GarageYrBlt'].value_counts()


# In[17]:

f = FillMissingValues('GarageYrBlt', alter_method='mean')
train1 = f.transform(train1)


# 查看GarageQual的分布，填充缺失值

# In[18]:

#--#
train1['GarageQual'].value_counts()


# In[19]:

f = FillMissingValues('GarageQual', alter_method='ud', user_defined='TA')
train1 = f.transform(train1)


# 查看BsmtExposure的分布，填充缺失值

# In[20]:

#--#
train1['BsmtExposure'].value_counts()


# In[21]:

f = FillMissingValues('BsmtExposure', alter_method='ud', user_defined='No')
train1 = f.transform(train1)


# 查看BsmtFinType2的分布，填充缺失值

# In[22]:

#--#
train['BsmtFinType2'].value_counts()


# In[23]:

f = FillMissingValues('BsmtFinType2', alter_method='ud', user_defined='Unf')
train1 = f.transform(train1)


# 查看BsmtFinType1的分布，填充缺失值

# In[24]:

#--#
train['BsmtFinType1'].value_counts()


# In[25]:

f = FillMissingValues('BsmtFinType1', alter_method='ud', user_defined='Unf')
train1 = f.transform(train1)


# 查看BsmtCond的分布，填充缺失值

# In[26]:

#--#
train1['BsmtCond'].value_counts()


# In[27]:

f = FillMissingValues('BsmtCond', alter_method='ud', user_defined='TA')
train1 = f.transform(train1)


# 查看BsmtQual的分布，填充缺失值

# In[28]:

#--#
train1['BsmtQual'].value_counts()


# In[29]:

f = FillMissingValues('BsmtQual', alter_method='ud', user_defined='TA')
train1 = f.transform(train1)


# 查看MasVnrArea的分布，填充缺失值

# In[30]:

#--#
#train1['MasVnrArea'].value_counts()


# In[31]:

f = FillMissingValues('MasVnrArea', alter_method='mean')
train1 = f.transform(train1)


# 查看MasVnrType的分布，填充缺失值

# In[32]:

#--#
train1['MasVnrType'].value_counts()


# In[33]:

f = FillMissingValues('MasVnrType', alter_method='ud', user_defined='None')
train1 = f.transform(train1)


# 查看Electrical的分布，填充缺失值

# In[34]:

#--#
train1['Electrical'].value_counts()


# In[35]:

f = FillMissingValues('Electrical', alter_method='ud', user_defined='SBrkr')
train1 = f.transform(train1)


# ### Feature 分成数值型和字符串型

# In[36]:

#train1.dtypes


# In[37]:

numeric_feats = train1.dtypes[train1.dtypes != 'object'].index


# In[38]:

#--#
numeric_feats


# In[39]:

numeric_feats = train1[list(numeric_feats)]


# In[40]:

cat_feats = train1.dtypes[train1.dtypes == 'object'].index


# In[41]:

cat_feats = train1[list(cat_feats)]


# In[42]:

#--#
cat_feats.columns


# ### 字符串型Feature进行LabelEncoding

# In[43]:

from feature_engineer.labelencoding import LabelEncoding
le = LabelEncoding(featureColNames=list(cat_feats.columns))
cat_feats2 = le.transform(cat_feats)


# 将转化后的cat_feats2和numeric_feats合并成train2

# In[44]:

train2 = pd.concat([numeric_feats, cat_feats2],axis=1)


# In[55]:

train2_copy = train2.copy()


# ### 特征选取

# In[45]:

from feature_engineer.featureselect import FeatureSelect
fs = FeatureSelect(featureColNames = list(train2.columns), topN=20, labelColName='SalePrice',discretedMethod=0 , selectMethod='ginigain')
result = fs.transform(train2)


# In[46]:

train3 = train2[['LotArea','GrLivArea','BsmtUnfSF','1stFlrSF','TotalBsmtSF','BsmtFinSF1','GarageArea','2ndFlrSF','MasVnrArea','WoodDeckSF','OpenPorchSF',
                'BsmtFinSF2','EnclosedPorch','YearBuilt','GarageYrBlt','ScreenPorch','YearRemodAdd','Neighborhood','LowQualFinSF','MiscVal','3SsnPorch',
                'Exterior2nd','MSSubClass','Exterior1st','TotRmsAbvGrd','SalePrice']]


# 根据各特征列信息增益的排序大小，最终输入模型的特征列选取了排在前25的特征列和标签列。

# In[47]:

#--#
train3.head()


# In[56]:

train3_copy = train3.copy()


# ### splitting train and test data

# In[48]:

from setoperation.Split import Split
s =  Split(fraction=0.2)                                                                                                                                                                                                                                                              
train, test = s.transform(train3)


# ### Linear Regression

# In[49]:

from ml.regression import LinearRegression
lr = LinearRegression(labelColName='SalePrice', featureColNames=list(train3.drop('SalePrice',1)))
lr.train(train)
predict = lr.transform(test)


# In[50]:

#--#
predict.head()


# ### GBDT Regression

# In[51]:

from ml.regression import GBDTRegression
gr = GBDTRegression(labelColName='SalePrice', featureColNames=list(train3.drop('SalePrice',1)), loss='ls',treeNum=500,learnRate=0.05,maxDeep=10)
gr.train(train)
predict2 = gr.transform(test)


# In[52]:

#--#
predict2.head()


# In[53]:

gr

