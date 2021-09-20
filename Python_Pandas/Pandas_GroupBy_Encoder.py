import os
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)
train_T = df['Survived']
df = df.drop(['PassengerId','Survived'],axis=1)
print(df.head())
#取船票票號(Ticket),乘客年齡(Age)做群聚編碼
df['Ticket'] = df['Ticket'].fillna('None')
df['Age'] = df['Age'].fillna(df['Age'].mean())

mean_df = df.groupby(['Ticket'])['Age'].mean().reset_index()
mode_df = df.groupby(['Ticket'])['Age'].apply(lambda x:x.mode()[0]).reset_index()
median_df  = df.groupby(['Ticket'])['Age'].median().reset_index()
max_df = df.groupby(['Ticket'])['Age'].max().reset_index()
min_df = df.groupby(['Ticket'])['Age'].min().reset_index()

temp = pd.merge(mean_df,mode_df,how='left',on=['Ticket'])
temp = pd.merge(temp,median_df,how='left',on=['Ticket'])
temp = pd.merge(temp,max_df,how='left',on=['Ticket'])
temp = pd.merge(temp,min_df,how='left',on=['Ticket'])
temp.columns = ['Ticket','Age_Mean','Age_Mode','Age_Median','Age_Max','Age_Min']
print(temp.head())

df = pd.merge(df,temp,how='left',on=['Ticket'])
df = df.drop(['Ticket'],axis=1)
print(df.head())
#取 int64、float64 兩種數值欄位，存於num_features中
num_features = []
for dtype,featrue in zip(df.dtypes,df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(featrue)
print('int64 and float64 num_featrues:{}'.format(num_features))
df = df[num_features]
df = df.fillna(-1)
print(df.head())
MME = MinMaxScaler()
#temp組
df_minus = df.drop(['Age_Mean', 'Age_Mode', 'Age_Median', 'Age_Max', 'Age_Min'],axis=1)
# 原始特徵 + 邏輯斯
train = MME.fit_transform(df_minus)
est = LogisticRegression()
score = cross_val_score(est,train,train_T,cv=5).mean()
print('No groupby_data:',score)
train_X = MME.fit_transform(df)
score_1 = cross_val_score(est,train_X,train_T,cv=5).mean()
print('Have groupby_data:',score_1)