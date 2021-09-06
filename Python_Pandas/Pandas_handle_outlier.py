import os
import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import isnull
from pandas.io.parsers import read_csv
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
#計算Q0-Q100
'''
all_q  = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]
        ['AMT_ANNUITY'], q = i) for i in range(100)]
print('所有的Q\n',all_q)
'''
print('Before replace NAs,number of row the AMT_ANNUITY is NAs:{}'.format(sum(app_train['AMT_ANNUITY'].isnull())))
q_50 = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],50)
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50
print('After replace NAs,number of row the AMT_ANNUITY is NAs:{}'.format(sum(app_train['AMT_ANNUITY'].isnull())))
print('==original==')
print(app_train['AMT_ANNUITY'].describe())

def normalize_value(x):
    x = (((x - min(x))/(max(x) - min(x))) - 0.5) * 2
    return x
app_train['AMT_ANNUITY_NOR'] = normalize_value(app_train['AMT_ANNUITY'])
print('== Normalized data ==')
print(app_train['AMT_ANNUITY_NOR'].describe())

print('Before replace NAs,number of row the AMT_GOOD_PRICE  is NAs:{}'.format(sum(app_train['AMT_GOODS_PRICE'].isnull())))

print('Mode:',app_train['AMT_GOODS_PRICE'].value_counts().head())
Mode = app_train['AMT_GOODS_PRICE'].value_counts().head(1).values
print('Mode:',Mode)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(),'AMT_GOODS_PRICE'] = Mode[0]
print('After replace NAs,number of row the AMT_GOOD_PRICE  is NAs:{}'.format(sum(app_train['AMT_GOODS_PRICE'].isnull())))