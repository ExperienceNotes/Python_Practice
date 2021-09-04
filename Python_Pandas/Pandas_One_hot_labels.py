import os
import pandas as pd
dir = './Datasets/'
path = os.path.join(dir,'application_train.csv')
app_train = pd.read_csv(path)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
print(sub_train.loc[:10])
print('---------------------------------------------')
sub_train = pd.get_dummies(sub_train)#one-hot encoder
print(sub_train.shape)
print(sub_train.head())