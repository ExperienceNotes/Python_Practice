import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
cut_relu = [-0.1,0,2,5,np.inf]
app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'],cut_relu,include_lowest=True)
print(app_train['CNT_CHILDREN_GROUP'].value_counts())

grp = ['CNT_CHILDREN_GROUP','TARGET']
group_pd = app_train.groupby(grp)['AMT_INCOME_TOTAL']
print(group_pd.mean())

plt_column = 'AMT_INCOME_TOTAL'
plt_by = ['CNT_CHILDREN_GROUP', 'TARGET']
app_train.boxplot(column=plt_column, by = plt_by, showfliers = False, figsize=(12,12))
plt.suptitle('AMT_INCOME_TOTAL')
plt.show()

app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] =group_pd.apply(lambda x :(x-np.mean(x))/np.std(x))
print(app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'])