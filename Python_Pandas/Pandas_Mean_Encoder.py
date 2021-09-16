import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import copy,time
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

dir = './Datasets/'
path_tr = os.path.join(dir,'titanic_train.csv')
path_te = os.path.join(dir,'titanic_test.csv')
train_df = pd.read_csv(path_tr)
test_df = pd.read_csv(path_te)
train_T = train_df['Survived']
ids = test_df['PassengerId']
train_df = train_df.drop(['Survived','PassengerId'],axis=1)
test_df = test_df.drop(['PassengerId'],axis=1)
df = pd.concat([train_df,test_df])

object_feature = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'object':
        object_feature.append(feature)
print('object_feature:{}'.format(object_feature))
df = df[object_feature]
df = df.fillna('None')
train_nums = train_T.shape[0]
print(df.head())
#對照組:LabelE + LogisticRegression
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_temp = df_temp[:train_nums]
train_temp = train_temp.drop(['Name'],axis=1)#個人覺得id不會影響存活率
est = LogisticRegression()
start = time.time()
print("train_temp_shape:{}".format(train_temp.shape))
crv_temp = cross_val_score(est,train_temp,train_T,cv=5).mean()
print("cross_val_score_temp:{}".format(crv_temp))
print("time:{}sec".format(time.time() - start))
print(train_temp.nunique())
#Mean_Label + LogisticRegression
data = pd.concat([df[:train_nums],train_T],axis=1)
for c in df.columns:
    #重點!
    mean_df = data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']#!!! 差一個空白差的天差地遠....mean_df.columns = [c,f'{c}_mean']
    data = pd.merge(data,mean_df,on=c,how='left')
    data = data.drop([c],axis=1)
data = data.drop(['Survived', 'Name_mean', 'Ticket_mean'] , axis=1)
est = LogisticRegression()
start = time.time()
crv_mean = cross_val_score(est,data,train_T,cv=5).mean()
print("cross_val_score_Mean_Label:{}".format(crv_mean))
print("time:{}sec".format(time.time() - start))