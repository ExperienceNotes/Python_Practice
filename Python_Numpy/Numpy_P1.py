import numpy as np
print('numpy_version:',np.__version__)
arr = np.arange(10)
arr = arr[arr%2 ==1]
print(arr)
array_bool = np.full((3,3),True,dtype=bool)
print('3*3_numpy_array:')
print(array_bool)
array_replace = np.arange(10)
array_replace[array_replace%2 == 1] = -1
print("array_replace:",array_replace)
array_where = np.arange(10)
array_where = np.where(array_where%2 == 1,-1,array_where)
print("array_where:",array_where)