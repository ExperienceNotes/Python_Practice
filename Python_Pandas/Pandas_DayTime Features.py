import os
import datetime
import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import Scalar
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

dir = './Datasets/'
path = os.path.join(dir,'taxi_data1.csv')
df = pd.read_csv(path)
train_T = df['fare_amount']
df = df.drop(['fare_amount'],axis = 1)
print(df.head())
#時間特徵分解方式:使用datatime

df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S UTC'))
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x,'%Y')).astype('int64')
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x,'%m')).astype('int64')
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x,'%H')).astype('int64')
df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x,'%M')).astype('int64')
df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x,'%S')).astype('int64')
print(df.head())

df_temp = df.drop(['pickup_datetime'],axis=1)
MMS = MinMaxScaler()
train = MMS.fit_transform(df_temp)
Liner = LinearRegression()
print('Liner Reg Score :{}'.format(cross_val_score(Liner,train,train_T,cv=5).mean()))
GDBT = GradientBoostingRegressor()
print('Gradient Boost Reg Score :{}'.format(cross_val_score(GDBT,train,train_T,cv=5).mean()))

df['pickup_dow'] = df['pickup_datetime'].apply(lambda x : datetime.datetime.strftime(x,'%w')).astype('int64')
df['pickup_woy'] = df['pickup_datetime'].apply(lambda x : datetime.datetime.strftime(x,'%W')).astype('int64')
print(df.head())
df_temp = df.drop(['pickup_datetime'],axis=1)
train = MMS.fit_transform(df_temp)
print('Liner Reg Score :{}'.format(cross_val_score(Liner,train,train_T,cv=5).mean()))
print('Gradient Boost Reg Score :{}'.format(cross_val_score(GDBT,train,train_T,cv=5).mean()))

# 加上"日週期"特徵 (參考講義"週期循環特徵")****重點****
import math
df['day_cycle'] = df['pickup_hour']/12 + df['pickup_minute']/720 + df['pickup_second']/43200
df['day_cycle'] = df['day_cycle'].map(lambda x:math.sin(x*math.pi))
print(df.head())
df_temp = df.drop(['pickup_datetime'],axis=1)
train = MMS.fit_transform(df_temp)
print('Liner Reg Score :{}'.format(cross_val_score(Liner,train,train_T,cv=5).mean()))
print('Gradient Boost Reg Score :{}'.format(cross_val_score(GDBT,train,train_T,cv=5).mean()))
# 加上"年週期"與"周週期"
df['year_cycle'] = df['pickup_month']/6 +df['pickup_day']/180
df['year_cycle'] = df['year_cycle'].map(lambda x:math.cos(x*math.pi))
df['week_cycle'] = df['pickup_dow']/3.5 + df['pickup_hour']/84
df['week_cycle'] = df['week_cycle'].map(lambda x:math.sin(x*math.pi))
print(df.head())

df_temp = df.drop(['pickup_datetime'],axis=1)
train = MMS.fit_transform(df_temp)
print('Liner Reg Score :{}'.format(cross_val_score(Liner,train,train_T,cv=5).mean()))
print('Gradient Boost Reg Score :{}'.format(cross_val_score(GDBT,train,train_T,cv=5).mean()))