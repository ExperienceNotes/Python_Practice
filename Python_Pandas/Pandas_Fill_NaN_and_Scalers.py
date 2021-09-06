import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
dir = './Datasets/'
path_Tr = os.path.join(dir,'titanic_train.csv')
path_Te = os.path.join(dir,'titanic_test.csv')
df_train = pd.read_csv(path_Tr)
df_test = pd.read_csv(path_Te)
train_L = df_train['Survived']
print('shape:',train_L.shape)
ids = df_test['PassengerId']
df_train = df_train.drop(['Survived','PassengerId'],axis = 1)
df_test = df_test.drop(['PassengerId'],axis = 1)
df = pd.concat([df_train,df_test])
print('df_shape:',df.shape)
#取出只有數值型欄位
num_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64' or dtype == 'float64':
        num_features.append(feature)
print('{} Feature:{}'.format(len(num_features),num_features))
df = df[num_features]
print(df.shape)#1309,5
num_train  = train_L.shape[0]
print(df.head(1))
# 空值補-1
df_m1 = df.fillna(-1)
train_X = df_m1[:num_train]
estimator = LogisticRegression()#評估羅吉斯回歸
cross_val_score(estimator,train_X,train_L,cv = 5).mean()
print('cross_val_score_-1:',cross_val_score(estimator,train_X,train_L,cv = 5).mean())
# 空值補0
df_m0 = df.fillna(0)
train_X = df_m0[:num_train]
estimator = LogisticRegression()#評估羅吉斯回歸
cross_val_score(estimator,train_X,train_L,cv = 5).mean()
print('cross_val_score_0:',cross_val_score(estimator,train_X,train_L,cv = 5).mean())
# 空值補平均值
df_mn = df.fillna(df.mean())
train_X = df_mn[:num_train]
estimator = LogisticRegression()#評估羅吉斯回歸
cross_val_score(estimator,train_X,train_L,cv = 5).mean()
print('cross_val_score_mean:',cross_val_score(estimator,train_X,train_L,cv = 5).mean())
#以上判斷補0優秀一點
#使用MinMaxScaler(最小最大化)
df_temp = MinMaxScaler().fit_transform(df_mn)
train_X = df_temp[:num_train]
estimator = LogisticRegression()
score = cross_val_score(estimator,train_X,train_L,cv = 5).mean()
print('cross_val_score_0+MinMaxScaler:',score)
#使用StandardScaler(標準化)
df_temp = StandardScaler().fit_transform(df_mn)
train_X = df_temp[:num_train]
estimator = LogisticRegression()
score = cross_val_score(estimator,train_X,train_L,cv = 5).mean()
print('cross_val_score_0+StandardScaler:',score)
#搭配cross_val_score_0+MinMaxScaler 最佳~