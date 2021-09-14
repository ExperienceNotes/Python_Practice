import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
dir = './Datasets/'
path_tr = os.path.join(dir,'titanic_train.csv')
path_te = os.path.join(dir,'titanic_test.csv')
train_df = pd.read_csv(path_tr)
test_df = pd.read_csv(path_te)
print('train_shape',train_df.shape)
train_T = train_df['Survived']
ids = test_df['PassengerId']
train_df = train_df.drop(['Survived','PassengerId'],axis=1)
test_df = test_df.drop(['PassengerId'],axis = 1)
df = pd.concat([train_df,test_df])
print('shape_concat:',df.shape)
print(df.head())

#重點 Featrue_Enginnering
LabE = LabelEncoder()
MMS = MinMaxScaler()

for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LabE.fit_transform(list(df[c].values))
        #e.q['male' 'female' 'female' ... 'male' 'male' 'male']
    df[c] = MMS.fit_transform(df[c].values.reshape(-1,1))
#reshape(-1,1) 會變成 row(n,1) #reshape(1,-1) 會變成 row(1,n)
print('Feature_after_shape:',df.shape)
print(df.head())
#---------------------------------------------------
train_num = train_T.shape[0]
train = df[:train_num]
test = df[train_num:]
est = LogisticRegression()
est.fit(train,train_T)
pred = est.predict(test)
'''
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv('titanic_baseline.csv', index=False)
'''