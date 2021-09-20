import os
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

dir = './Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)
train_T = df['Survived']
df = df.drop(['PassengerId','Survived'],axis=1)
print(df.head())