import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
labelE = LabelEncoder()
print(app_train.shape)
print(app_train.head(2))

for col in app_train:
    if app_train[col].dtype == 'object':
        #如果只有兩種欄位
        if len(list(app_train[col].unique())) <= 2:
            #use LabelEncoder
            app_train[col] = labelE.fit_transform(app_train[col])
print(app_train.shape)
# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
#print('corr_T:\n',app_train.corr()['TARGET'])
correlations = app_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations:\n{}'.format(correlations.tail(15)))
print('Most Negative Correlations:\n{}'.format(correlations.head(15)))
#負相關
plt.scatter(app_train['EXT_SOURCE_3'],app_train['TARGET'])#看不出來
app_train.boxplot(by = 'TARGET',column='EXT_SOURCE_3')
plt.show()
#正相關，有夠爛...
plt.scatter(app_train['DAYS_EMPLOYED'],app_train['TARGET'])#看不出來
app_train.boxplot(by = 'TARGET',column='DAYS_EMPLOYED')
plt.show()