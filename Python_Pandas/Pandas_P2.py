import pandas as pd
import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
print(df.info)#columns 資訊
print(df.describe())#所有統計數據
print("show first 3 data::",df.iloc[:3])#0、1、2
#df.head(3) any_method
print(df.loc[:,['animal','age']])#取出animal、age 兩列的資料
#df[['animal','age']]
print(df.loc[df.index[[3,4,8]],['animal','age']])
print(df[df['age'] > 3])
print(df[df['age'].isnull()])#取出age值缺失的行
print(df[(df['age']>2) & (df['age']<4)])#取出age在2,4間的行
df.loc['f','age'] = 1.5
print('visits sum:',df['visits'].sum())#計算某col總和
df.loc['k'] = [5.5,'doggy','no',1]#插入
df = df.drop('k')#刪除
print(df['animal'].value_counts())
#先按age降序排列，後按visits升序排列
df.sort_values(by=['age', 'visits'], ascending=[False, True])
df['priority'] = df['priority'].map({'yes':True,'no':False})