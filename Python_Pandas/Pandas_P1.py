from datetime import date
from numpy import array
import numpy
import pandas as pd
import numpy as np
print(pd.__version__)
array = np.arange(5)
df = pd.Series(array)
print(df)
print('------------字典------------')
d = {'a':1,'b':2,'c':3,'d':4,'e':5}
df = pd.Series(d)
print(df)

date = pd.date_range('today',periods=6)
num_arr = np.random.rand(6,4)#隨機傳入陣列大小(6,4)
columns = ['A','B','C','D']
df1 = pd.DataFrame(num_arr,index = date,columns = columns)
print(df1)

data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(data, index=labels)
print(df2)