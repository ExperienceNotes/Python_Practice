from sklearn.datasets import load_boston,load_wine
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model._glm.glm import _y_pred_deviance_derivative
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

#讀取Boston資料
boston = load_boston()

#切分訓練集/測試集
x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.1,random_state=4)
#建立一個線性模型
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print("Mean squared error:{}".format(mean_squared_error(y_test,y_pred)))

#讀取Boston資料
wine = load_wine()
#切分資料集/測試集
x_train,x_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.1,random_state=4)
reg_l = LogisticRegression()
reg_l.fit(x_train,y_train)
y_pred = reg_l.predict(x_test)

acc = accuracy_score(y_test,y_pred)
print('Accuracy:{}'.format(acc))