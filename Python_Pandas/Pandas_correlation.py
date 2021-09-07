import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt
#弱相關、無相關
x = np.random.randint(0,50,1000)
y = np.random.randint(0,50,1000)
print(np.corrcoef(x,y))#相關矩陣
plt.scatter(x,y)
plt.show()
#強相關、很有關
x_1 = np.random.randint(0,50,1000)
y_1 = x_1 + np.random.randint(0,5,1000)
print(np.corrcoef(x_1,y_1))
plt.scatter(x_1,y_1)
plt.show()
#負相關、有關係但關係不好
x_2 = np.random.randint(0,50,1000)
y_2 = -x_2 + np.random.randint(0,10,1000)
print(np.corrcoef(x_2,y_2))
plt.scatter(x_2,y_2)
plt.show()