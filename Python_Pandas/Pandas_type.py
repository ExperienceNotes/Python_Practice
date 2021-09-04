import os
from numpy import dtype
import pandas as pd

dir = './Datasets/'
path_tr = os.path.join(dir,'titanic_train.csv')
path_te = os.path.join(dir,'titanic_test.csv')
titanic_Tr = pd.read_csv(path_tr)
titanic_Te = pd.read_csv(path_te)
print('titanic_Tr.shape:{}'.format(titanic_Tr.shape))
print('titanic_Te.shape:{}'.format(titanic_Te.shape))
train_Y = titanic_Tr['Survived']#活不活給答案
ids = titanic_Te['PassengerId']
titanic_Tr = titanic_Tr.drop(['Survived','PassengerId'],axis = 1)
titanic_Te = titanic_Te.drop(['PassengerId'],axis = 1)
df = pd.concat([titanic_Tr,titanic_Te])
print(df.shape)#891+418 = 1309
dtype_df = df.dtypes.reset_index()#資料型態的種類
dtype_df.columns = ['Count','Colume_Type']
print(dtype_df)
dtype_df = dtype_df.groupby('Colume_Type').aggregate('count').reset_index()
print(dtype_df)
int_features, float_featue, object_feature = [],[],[]
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64':
        int_features.append(feature)
    elif dtype == 'float64':
        float_featue.append(feature)
    else:
        object_feature.append(feature)
print('int_features:{}'.format(int_features))
print('float_featue:{}'.format(float_featue))
print('object_feature:{}'.format(object_feature))
print('object_feature_nunique:',df[object_feature].nunique())#相異值

data_type = {'int_features':int_features,
            'float_featue':float_featue,
            'object_feature':object_feature}
for d in data_type:
    if d != 'object_feature':
        print('\n{} mean():\n{}\n============'.format(d,df[data_type[d]].mean()))
        print('\n{} max():\n{}\n============'.format(d,df[data_type[d]].max()))
    print('\n{} nunique():\n{}\n============'.format(d,df[data_type[d]].nunique()))