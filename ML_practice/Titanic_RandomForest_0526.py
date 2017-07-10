
# coding: utf-8

# In[1]:

import os 
os.chdir(u'E:/量知/Ali/dmlib')


# # Titanic问题
数据集：
变量      定义                           值
Variable	Definition  	                   Key
survival	Survival	                      0 = No, 1 = Yes
pclass	   Ticket class	                     1 = 1st, 2 = 2nd, 3 = 3rd
sex	      Sex	
Age	      Age                            in years	
sibsp	   # of siblings / spouses aboard the Titanic  the number of siblings or spouses travelling with each                                         passenger.
parch	   # of parents / children aboard the Titanic	the number of parents or children travelling with each                                         passenger.
ticket	   Ticket number	
fare	   Passenger fare	                   how much each passenger paid for their journey.
cabin	   Cabin number	
embarked	Port of Embarkation	                 C = Cherbourg, Q = Queenstown, S = Southampton
# In[2]:

import pandas as pd
train_data = pd.read_csv(u'E:/量知/ML/DATA/Titanic/train.csv')
test_data = pd.read_csv(u'E:/量知/ML/DATA/Titanic/test.csv')


# In[3]:

train_data.head()


# In[4]:

train_data.describe()


# # 观察数据

# ## 特征的分布

# In[5]:

surv = train_data[train_data['Survived']==1]
nosurv = train_data[train_data['Survived']==0]
print('Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i'      %(len(surv), 1.0*len(surv)/len(train_data)*100.0, len(nosurv), 1.0*len(nosurv)/len(train_data)*100.0,       len(train_data)))


# In[6]:

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[7]:

get_ipython().magic(u'matplotlib inline')
#在jupyter notebook作图需要加这个命令
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins = range(0, 81, 1), kde=False, color='blue')
sns.distplot(nosurv['Age'].dropna().values, bins = range(0, 81, 1), kde=False, color='red',axlabel='Age')
plt.subplot(332)
sns.barplot('Sex','Survived',data=train_data,ci=0)
plt.subplot(333)
sns.barplot('Pclass','Survived',data=train_data)
plt.subplot(334)
sns.barplot('SibSp','Survived',data=train_data)
plt.subplot(335)
sns.barplot('Parch','Survived',data=train_data)
plt.subplot(336)
sns.barplot('Embarked','Survived',data=train_data)
plt.subplot(337)
sns.distplot(surv['Fare'].dropna().values,bins = range(0,513,1),kde=False, color='blue')
sns.distplot(nosurv['Fare'].dropna().values,bins = range(0,513,1),kde=False, color='red', axlabel='Fare')
plt.subplot(338)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color='blue')
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color='red',axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
import numpy as np
print("Median age survivors: %.1f, Median age non-survivers: %.1f"     %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))


# 上图可视化了不同特征的分布情况，数值型特征用直方图表示，类别型特征用柱状图表示。
# 
# 从上图可以看出：年龄在10岁以下的孩子存活率较高，女性比男性存活率高，1等舱乘客比其他乘客存活率高，有1-3个亲人的乘客比独自一人或全家人出行的乘客存活率更高，出发港口为C的乘客存活率较高，船舱越便宜的乘客存活率越小。

# 柱状图中的黑色竖线是置信度线，显示为ci=1,不显示为ci=0,长度表示置信区间的大小，画大约估计的值。

# ## 特征之间的关系

# In[8]:

plt.figure(figsize=(12,10))
foo = sns.heatmap(train_data.drop(['PassengerId','Name'],axis=1).corr(),vmax=0.4,square=True, annot=True)


# 从图中可以看出，pclass，fare与survived有很密切的关系.

# # 数据预处理

# ### Missing Values

# In[9]:

print(train_data.isnull().sum())


# In[10]:

print(test_data.isnull().sum())


# #### Embarked

# In[11]:

print(train_data[train_data['Embarked'].isnull()])


# 因为这两名乘客都是女性，且在1等舱，而且幸存，由幸存人员embarked分布可以将其填为‘C’

# In[12]:

#train_data['Embarked'].iloc[61]='C'
#train_data['Embarked'].iloc[829]='C'
from preprocess.FillMissingValues import FillMissingValues
FillMissingValues('Embarked',alter_method='ud',user_defined='C')


# In[13]:

print (train_data.iloc[61])


# #### Fare

# In[14]:

print(test_data[test_data['Fare'].isnull()])


# In[15]:

combine = pd.concat([train_data.drop('Survived',1),test_data])
test_data['Fare'].iloc[152] = combine['Fare'][combine['Pclass'] == 3].dropna().median()
print(test_data['Fare'].iloc[152])


# # 特征工程

# 生成新的特征列

# In[16]:

train = train_data
test = test_data
train['Child'] = train['Age']<=10
train['Young'] = (train['Age']>=18) & (train['Age']<=40)
train['Young_m'] = (train['Age']>=18) & (train['Age']<=40) & (train['Sex']=="male")
train['Young_f'] = (train['Age']>=18) & (train['Age']<=40) & (train['Sex']=="female")
train['Cabin_known'] = train['Cabin'].isnull() == False
train['Age_known'] = train['Age'].isnull() == False
train['Family'] = train['SibSp'] + train['Parch']
train['Alone']  = (train['SibSp'] + train['Parch']) == 0
train['Large Family'] = (train['SibSp']>2) | (train['Parch']>3)
train['Deck'] = train['Cabin'].str[0]                        #船舱号码的第一位
train['Deck'] = train['Deck'].fillna(value='U')
train['Ttype'] = train['Ticket'].str[0]                      #船票号码的第一位
train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


test['Child'] = test['Age']<=10
test['Young'] = (test['Age']>=18) & (test['Age']<=40)
test['Young_m'] = (test['Age']>=18) & (test['Age']<=40) & (test['Sex']=="male")
test['Young_f'] = (test['Age']>=18) & (test['Age']<=40) & (test['Sex']=="female")
test['Cabin_known'] = test['Cabin'].isnull() == False
test['Age_known'] = test['Age'].isnull() == False
test['Family'] = test['SibSp'] + test['Parch']
test['Alone']  = (test['SibSp'] + test['Parch']) == 0
test['Large Family'] = (test['SibSp']>2) | (test['Parch']>3)
test['Deck'] = test['Cabin'].str[0]
test['Deck'] = test['Deck'].fillna(value='U')
test['Ttype'] = test['Ticket'].str[0]
test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[17]:

train['Title'].value_counts()


# In[18]:

train2 = train[train['Title'].isin(['Mr','Miss','Mrs','Master'])]
foo = train2['Age'].hist(by=train2['Title'], bins=np.arange(0,81,1))


# 根据称呼确定年龄范围：

# In[19]:

train['Young'] = (train['Age']<=30) | (train['Title'].isin(['Master','Miss','Mlle','Mme']))
test['Young'] = (test['Age']<=30) | (test['Title'].isin(['Master','Miss','Mlle','Mme']))


# 特征离散

# In[20]:

train["Sex"] = train["Sex"].astype("category")
train["Sex"].cat.categories = [0,1]
train["Sex"] = train["Sex"].astype("int")
train["Embarked"] = train["Embarked"].astype("category")
train["Embarked"].cat.categories = [0,1,2]
train["Embarked"] = train["Embarked"].astype("int")
train["Deck"] = train["Deck"].astype("category")
train["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
train["Deck"] = train["Deck"].astype("int")

test["Sex"] = test["Sex"].astype("category")
test["Sex"].cat.categories = [0,1]
test["Sex"] = test["Sex"].astype("int")
test["Embarked"] = test["Embarked"].astype("category")
test["Embarked"].cat.categories = [0,1,2]
test["Embarked"] = test["Embarked"].astype("int")
test["Deck"] = test["Deck"].astype("category")
test["Deck"].cat.categories = [0,1,2,3,4,5,6,7]
test["Deck"] = test["Deck"].astype("int")


# In[21]:

ax = plt.subplots( figsize =( 12 , 10 ) )
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=1.0, square=True, annot=True)


# In[22]:

print(train.isnull().sum())


# In[23]:

train.columns


# In[24]:

train = train.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Cabin','Age_known','Family','Title','Fare','Ttype'],axis=1)


# # 构建模型

# In[25]:

from sklearn.model_selection import train_test_split
training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"     %(train.shape[0],training.shape[0],testing.shape[0]))


# In[26]:

y_test = testing['Survived']
testing = testing.drop('Survived',1)


# In[27]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(training.drop('Survived',1),training['Survived'])
predict1 = clf.predict(testing)
predict1


# In[28]:

pd.Series(predict1).value_counts()


# In[29]:

from ml.classify import RandomForest
from pipeline import Pipeline
from feature_engineer.randomForestImportance import RandomForestImportance
alg1 = RandomForest(labelColName='Survived', featureColNames=list(testing.columns),treeNum=100,maxTreeDeep=None,randomColNum='log2',minNumObj=3)
imp = RandomForestImportance(alg1, labelColName='Survived')
p1 = Pipeline([alg1])
model1 = p1.fit(training)
#predict = model1.transform(testing.drop('Survived',1))
predict = model1.transform(testing)


# In[30]:

predict.head()


# In[31]:

predict.label.value_counts()


# ### 交叉验证

# In[32]:

from sklearn.model_selection import cross_val_score
print alg1.n_estimators
print alg1
score = cross_val_score(alg1, train.drop('Survived',1), train['Survived'], cv=5)  # k折交叉检验


# In[33]:

score


# In[34]:

score.mean()


# ### 调参

# In[35]:

# use a grid search algorithm to find the best parameters to run the classifier.
from sklearn.model_selection import GridSearchCV
param_grid = { 'treeNum':[50,100,300,500,700,1000],
               'minNumObj':[1,3,5,10,30],
               'randomColNum':['auto','sqrt','log2']}   # max_features=n_features.
gs = GridSearchCV(estimator=alg1, param_grid=param_grid, scoring='accuracy', cv=5)
gs = gs.fit(train.drop('Survived',1), train['Survived'])
print gs.best_score_
print gs.best_params_


# ### 混淆矩阵

# In[36]:

from sklearn.metrics import confusion_matrix
predict['Survived'] = y_test
x = confusion_matrix(predict['Survived'], predict['label'],labels=[0,1] )
x


# In[37]:

from evaluate.ConfusionMatrix import ConfusionMatrix
predict['Survived'] = y_test
cm = ConfusionMatrix(labelColName='Survived',predictionColName='label')  #y_true,y_predict
cmatrix = cm.transform(predict)


# In[38]:

cmatrix


# In[39]:

class_names = ['Survived','UnSurvived']
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


# 从图中可以看出：有8个人实际幸存了但被预测为死亡，有22个人死亡但被预测为幸存，其他为预测正确的人数。

# In[40]:

from evaluate.ClassificationEvaluate import ClassificationEvaluate
ce = ClassificationEvaluate(labelColName='Survived',scoreColName='predict_score')
ce.transform(predict)


# fpr-横坐标，rpr-纵坐标，fpr越小越好，tpr越大越好；AUC为ROC曲线下面的面积

# In[ ]:



