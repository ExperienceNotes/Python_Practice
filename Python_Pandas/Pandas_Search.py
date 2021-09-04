import os
import pandas as pd
from pandas.core.frame import DataFrame

dir =  './Datasets/'
path = os.path.join(dir,'application_train.csv')
print('Path of read in data: %s' % (path))
app_train = pd.read_csv(path)
print(app_train.shape)#307511 , 122
print(app_train.info())#資料資訊
print(app_train.describe())#所有欄位統計數據
print(app_train.columns)#顯示有多少欄位
print(app_train.head())#不設定擷取前5比
print(app_train.iloc[0:5,0:2])#5 row , 2 colums
print(app_train.iloc[[1,3]])#指定1,3的欄位
print(app_train.loc[0:5,'SK_ID_CURR':'TARGET'])
print(app_train[['SK_ID_CURR','AMT_ANNUITY','AMT_CREDIT']])