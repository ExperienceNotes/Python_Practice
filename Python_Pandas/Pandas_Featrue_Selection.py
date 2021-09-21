import os
import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression,Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import compress
dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)

train_T = df['Survived']
df = df.drop(['PassengerId'],axis=1)
print(df.head())

corr = df.corr()
sns.heatmap(corr,cmap="YlGnBu")
plt.show()
df = df.drop(['Survived'],axis=1)
nums_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64' or dtype == 'float64':
        nums_features.append(feature)
print('nums_feature:{}'.format(nums_features))

df = df[nums_features]
df = df.fillna(-1)
print(df.head())
#原始特徵 + Logline
MME = MinMaxScaler()
train_Log = MME.fit_transform(df)
est_Log = LogisticRegression()
socre_Log = cross_val_score(est_Log,train_Log,train_T,cv=5).mean()
print('socre_Log:{}'.format(socre_Log))
#篩選相關係數大於0.05或小於-0.05的特徵
high_list = list(corr[(corr['Survived']>0.05)|(corr['Survived']<-0.05)].index)
high_list.pop(0)
print(high_list)
train_high_feature = MME.fit_transform(df[high_list])
socre_high_feature = cross_val_score(est_Log,train_high_feature,train_T,cv=5).mean()
print('socre_high_feature:{}'.format(socre_high_feature))

#使用L1 Embedding 做特徵選擇(自訂門檻)
#Note:L1 Embedding 扔需調整alpha且沒有一定的法則，所以並非好用的特徵選擇方式
L1_Reg = Lasso(alpha=0.005)
train_L1 = MME.fit_transform(df)
L1_Reg.fit(train_L1,train_T)
print('L1_Reg:{}'.format(L1_Reg.coef_))

L1_mask = list((L1_Reg.coef_>0)|(L1_Reg.coef_<0))
print('L1_mask:{}'.format(L1_mask))
L1_list = list(compress(list(df),list(L1_mask)))
print('L1_list:{}'.format(L1_list))
#L1_Enbedding + LogLine
train_L1_x = MME.fit_transform(df[L1_list])
scroe_L1_feature = cross_val_score(est_Log,train_L1_x,train_T,cv=5).mean()
print('scroe_L1_feature:{}'.format(scroe_L1_feature))