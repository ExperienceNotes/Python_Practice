import os
import pandas as pd
import matplotlib.pyplot as plt
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
print(app_train.describe())#看不出來哪個數值重要
int_features, float_features, object_features = [],[],[]
for dtype,feature in zip(app_train.dtypes,app_train.columns):
    if dtype == 'int64':
        int_features.append(feature)
    elif dtype == 'float64':
        float_features.append(feature)
    else:
        object_features.append(feature)
print('int_features:')
print(app_train[int_features].describe())#一樣難看
print(app_train[int_features].std())
print('float_features:')
print(app_train[float_features].describe())
print(app_train[float_features].std())
print('object_features:')
print(app_train[object_features].describe())
#AMT_ANNUITY calculate mean、std
print('AMT_ANNUITY_mean:{}'.format(app_train['AMT_ANNUITY'].mean()))
print('AMT_ANNUITY_std:{}'.format(app_train['AMT_ANNUITY'].std()))
#LIVE_CITY_NOT_WORK_CITY
#app_train['LIVE_CITY_NOT_WORK_CITY'].plot.hist(alpha = 0.6)
#plt.show()#這資料母湯
#OWN_CAR_AGE
plt.hist(app_train['OWN_CAR_AGE'])
plt.title('OWN_CAR_AGE')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
#計算數量
plt.bar(['F', 'M', 'XNA'], app_train['CODE_GENDER'].value_counts())
plt.title('CODE GENDER')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()