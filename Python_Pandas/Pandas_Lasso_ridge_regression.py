from sys import prefix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_diabetes

#讀取糖尿病資料集
diabetes = load_diabetes()
#切分訓練集/測試集
x_train,x_test,y_train,y_test = train_test_split(diabetes.data,diabetes.target,test_size=0.2,random_state=4)
#建立一個回歸模型
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print("reg.coef_:",reg.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
#建立一個套索回歸
reg_La = Lasso(alpha=0.3)
reg_La.fit(x_train,y_train)
y_pred_La = reg_La.predict(x_test)
print("reg_La.coef_:",reg_La.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_La))
#建立一個Ridge回歸
reg_Ri = Ridge(alpha=1.0)
reg_Ri.fit(x_train,y_train)
y_pred_Ri = reg_Ri.predict(x_test)
print("reg_Ri.coef_:",reg_Ri.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_Ri))