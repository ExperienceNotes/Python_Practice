import os
import pandas as pd
import numpy as bp
import copy
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)