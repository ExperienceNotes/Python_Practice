import os
from numpy.lib.polynomial import poly
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
LabE = LabelEncoder()
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)

for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            app_train[col] = LabE.fit_transform(app_train[col])
print('app_train_shape{}'.format(app_train.shape))
app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243
app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

print("Correlation between TARGET and DAYS_BIRTH",
        app_train['DAYS_BIRTH'].corr(app_train['TARGET']))
print((app_train['DAYS_BIRTH']/365).describe())

age_data = app_train[['TARGET','DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

#cut_lin = #(70-20)/(11-1) = 5
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],bins = np.linspace(20,70,11))
print(age_data.head(5))
# 依照組別彙整年齡資料 "每一欄的平均值"
age_groups  = age_data.groupby('YEARS_BINNED').mean()
print(age_groups)
#print(len(age_groups.index)) 10
plt.figure(figsize = (8,8))
plt.bar(range(len(age_groups.index)),age_groups['TARGET'])
plt.xticks(range(len(age_groups.index)),age_groups.index,rotation = 75)
plt.xlabel('Age_Group(years)')
plt.ylabel('Average Failure to Repay')
plt.title('Failture to Repay by Age Group')
plt.show()