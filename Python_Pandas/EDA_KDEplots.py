import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 # # day-age to year-age
#(307511, 3)
bin_cut = np.linspace(20,70,11)#切11個點得到10組
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],bins = bin_cut)
#顯示每組的量
print(age_data['YEARS_BINNED'].value_counts())
print(age_data.head())

year_group_sorted = np.sort(age_data['YEARS_BINNED'].unique())
plt.figure(figsize=(8,6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i])&
                (age_data['TARGET'] == 0),'YEARS_BIRTH'],label = str(year_group_sorted[i]))
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i])&
                (age_data['TARGET'] == 1),'YEARS_BIRTH'],label = str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()

# 計算每個年齡區間的 Target、DAYS_BIRTH與 YEARS_BIRTH 的平均值
age_groups  = age_data.groupby('YEARS_BINNED').mean()
plt.figure(figsize = (8, 8))
px = age_groups.index.astype(str)
py = 100 * age_groups['TARGET']
sns.barplot(px, py)
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
plt.show()