import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

dir = './Datasets/'
path_tr = os.path.join(dir,'titanic_train.csv')
path_te = os.path.join(dir,'titanic_test.csv')
train_df = pd.read_csv(path_tr)
test_df = pd.read_csv(path_te)
train_T = train_df['Survived']
ids = test_df['PassengerId']
train_df = train_df.drop(['Survived','PassengerId'],axis = 1)
test_df = test_df.drop(['PassengerId'],axis = 1)
df = pd.concat([train_df,test_df])
print(df.head())

num_feature = []

for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_feature.append(feature)
print('feature_nums:{}'.format(num_feature))

df = df[num_feature]
df = df.fillna(0)
MME = MinMaxScaler()
train_num = train_T.shape[0]
print(df.head())
#Fare 散布圖
sns.distplot(df['Fare'][:train_num])
plt.show()

df_MME = MME.fit_transform(df)
train = df_MME[:train_num]
est = LogisticRegression()
crv_mean = cross_val_score(est,train,train_T,cv = 5).mean()
print('cross_val_score_mean:',crv_mean)
#去偏化:log
df_log = copy.deepcopy(df)
df_log['Fare'] = np.log1p(df_log['Fare']+1)
sns.distplot(df_log['Fare'][:train_num])
plt.show()
df_log = MME.fit_transform(df_log)
train_log = df_log[:train_num]
est = LogisticRegression()
crv_mean_log = cross_val_score(est,train_log,train_T,cv=5).mean()
print('cross_val_score_log_mean:',crv_mean_log)
#boxcox
df_boxcox = copy.deepcopy(df)
df_boxcox['Fare'] = df_boxcox['Fare'] + 1
df_boxcox['Fare'] = stats.boxcox(df_boxcox['Fare'])[0]
sns.distplot(df_boxcox['Fare'][:train_num])
plt.show()
df_boxcox = MME.fit_transform(df_boxcox)
train_boxcox = df_boxcox[:train_num]
est = LogisticRegression()
crv_mean_boxcox = cross_val_score(est,train_boxcox,train_T,cv = 5).mean()
print('cross_val_score_boxcox_mean',crv_mean_boxcox)