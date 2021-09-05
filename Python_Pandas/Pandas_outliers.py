import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
#選出數值特徵
dtype_select = [np.dtype('int64'),np.dtype('float64')]
#檢視所有符合的數值
print(app_train.dtypes.isin(dtype_select))
#取出所有需要的特徵欄位
new_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])
filter_columns = app_train[new_columns].apply(lambda x:len(x.unique())!=2)
print('filter_columns:',filter_columns)
new_columns = list(app_train[new_columns].columns[filter_columns])
print('new_columns:',new_columns)#所有欄位
#慎重使用很多張圖
'''
for col in new_columns:
    fig = plt.figure(figsize=(18,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    app_train.boxplot(column = col, ax = ax1)
    app_train.hist(column = col, ax = ax2)
    plt.show()
'''
#AMT_INCOME_TOTAL: 客戶收入
#REGION_POPULATION_RELATIVE: 客戶居住地區的標準化人口（數字越大意味著客戶居住的地區人口越多
#OBS_60_CNT_SOCIAL_CIRCLE: 對客戶的社交環境進行多少次觀察，可觀察到 60 DPD（逾期天數）默認值
'''
for col_1 in ['AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 'OBS_60_CNT_SOCIAL_CIRCLE']:
    fig = plt.figure(figsize=(18,10))
    app_train.boxplot(column = col_1)
plt.show()
'''
#--------------------------------------------------------------------------------------------
'''
print('AMT_INCOME_TOTAL:','\n',app_train['AMT_INCOME_TOTAL'].describe())
## sort_index:根據index排序
## cumsum():累加總數
cdf = app_train['AMT_INCOME_TOTAL'].value_counts().sort_index().cumsum()
print('list(cdf.index):\n',list(cdf.index))
plt.plot(list(cdf.index),cdf/cdf.max())
plt.xlabel('AMT_INCOME_TOTAL')
plt.ylabel('ECDF')
## 限制x軸和y軸範圍
plt.xlim(cdf.index.min(), cdf.index.max() * 1.05)
plt.ylim([-0.05, 1.05])
plt.show()
plt.plot(np.log(list(cdf.index)), cdf / cdf.max())
plt.xlabel('Value (lof-scale)')
plt.ylabel('ECDF')
plt.ylim([-0.05, 1.05])
plt.show()
'''
#--------------------------------------------------------------------------------------------
'''
print('REGION_POPULATION_RELATIVE:\n',app_train['REGION_POPULATION_RELATIVE'].describe())
cdf_RE = app_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index().cumsum()
plt.plot(list(cdf_RE.index), cdf_RE/cdf_RE.max())
plt.xlabel('REGION_POPULATION_RELATIVE')
plt.ylabel('ECDF')
plt.xlim(cdf_RE.index.min(), cdf_RE.index.max() * 1.05)
plt.ylim([-0.05, 1.05])
plt.show()
app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()
print(app_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index())
# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏
'''
#--------------------------------------------------------------------------------------------
print('OBS_60_CNT_SOCIAL_CIRCLE\n',app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())
#有一個異常大的MAX
cdf_OB = app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index().cumsum()
plt.plot(list(cdf_OB.index),cdf_OB/cdf_OB.max())
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')
plt.ylabel('ECDF')
plt.xlim(cdf_OB.index.min()*0.95,cdf_OB.index.max()*1.05)
plt.ylim(-0.05,1.05)
plt.show()
app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()#左邊一大根~~~
#計算它的每一個欄位總數
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index())
data_OB = app_train['OBS_60_CNT_SOCIAL_CIRCLE'] <= 20
data_B = 'OBS_60_CNT_SOCIAL_CIRCLE'
app_train.loc[data_OB, data_B].hist()
plt.show()