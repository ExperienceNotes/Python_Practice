import numpy as np
from sklearn.model_selection import train_test_split
#模擬label不平衡 1只有40個
x = np.arange(1000).reshape(200,5)
y = np.zeros(200)
y[:40] = 1
# 選出 y 等於 1 的 index 與 y 等於 0 的 index
y_1_index,y_0_index = np.where(y==1),np.where(y==0)[0]
#將 (X, y) 中 y 等於 1 的部分取出作 train/test split，指定 test_size=10； y 等於 0 的部分也做一樣的操作
train_data_y1,test_data_y1,train_label_y1,test_label_y1 = train_test_split(x[y_1_index],y[y_1_index],test_size=10)
train_data_y0,test_data_y0,train_label_y0,test_label_y0 = train_test_split(x[y_0_index],y[y_0_index],test_size=10)
#再將分好的 data 與 label 合併起來，變回我們熟悉的 x_train, x_test, y_train, y_test
x_train, y_train = np.concatenate([train_data_y1, train_data_y0]), np.concatenate([train_label_y1, train_label_y0])
x_test, y_test = np.concatenate([test_data_y1, test_data_y0]), np.concatenate([test_label_y1, test_label_y0])
#此時y_test 的 label 就平衡了