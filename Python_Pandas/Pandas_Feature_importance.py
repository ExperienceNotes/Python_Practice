import os
import pandas as pd
import numpy as bp
import copy
from scipy.sparse import data
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)

train_T = df['Survived']
df = df.drop(['PassengerId','Survived'],axis=1)
print(df.head())
LabE = LabelEncoder()
MME = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LabE.fit_transform(list(df[c].values))
    df[c] = MME.fit_transform(df[c].values.reshape(-1,1))
print(df.head())
#隨機森林，將結果依照重要性由高到低排序
est = RandomForestClassifier()
est.fit(df.values,train_T)
feats = pd.Series(data = est.feature_importances_,index=df.columns)
feats = feats.sort_values(ascending=False)
print(feats)
#org_feature + RandomF
train_org = MME.fit_transform(df)
score_org = cross_val_score(est,train_org,train_T,cv=5).mean()
print('score_org:',score_org)
#高重要性特徵 + RandomF
high_feature = list(feats[:5].index)
train_high_f = MME.fit_transform(df[high_feature])
score_high_f = cross_val_score(est,train_high_f,train_T,cv=5).mean()
print('score_high_f:',score_high_f)
#觀察重要特徵與目標的分布:這個順序可能因為sklearn的版本不同有不同的結果
#1st:Sex
sns.violinplot(x = train_T,y=df['Sex'],fit_reg = False,scale = "width")
plt.show()
#2nd:Ticket
sns.violinplot(x = train_T,y = df['Ticket'],fit_reg = False,scale = 'width')
plt.show()
sns.violinplot(x = train_T,y = df['Cabin'],fit_reg = False,scale = 'width')
plt.show()
#製作特殊特徵:加、乘
df['Add_char'] = (df['Ticket']+df['Name'])/2
df['Multi_char'] = df['Ticket']*df['Name']
train_new_f = MME.fit_transform(df)
score_new_f = cross_val_score(est,train_new_f,train_T,cv=5).mean()
print('score_new_f:',score_new_f)