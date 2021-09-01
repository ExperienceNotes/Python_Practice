import numpy as np
from numpy.core.shape_base import vstack
array = np.arange(10)
array_reshape = np.reshape(array,(2,5))
print(array_reshape)
print("----------------------------------------")
a = np.arange(10)
a = np.reshape(a,(2,5))
b = np.repeat(1,10)#np.repeat(x,num) 重複x num次
b = np.reshape(b,(2,5))
#Method 1:
conca = np.concatenate([a,b],axis = 0)
print('concatenate:',conca)
#Method 2:
vstac = np.vstack([a,b])
print('vstac:',vstac)
print("----------------------------------------")
#Method 1:
conca_h = np.concatenate([a,b],axis = 1)
print('concatenate:',conca_h)
#Method 2:
hstac = np.hstack([a,b])
print('vstac:',hstac)
a_test = np.array([1,2,3])
a_repeat = np.repeat(a_test,3)
a_tile = np.tile(a_test,3)
conca_a = np.concatenate([a_repeat,a_tile],axis=0)
print('conca_a:',conca_a)