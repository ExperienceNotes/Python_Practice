import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#載入資料
iris = load_iris()
#資料切分
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=4)
#建立模型
clf = GradientBoostingClassifier()
#訓練模型
clf.fit(x_train,y_train)
#模型預測
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy:',acc)
#開始玩其他功課摟~~~
digits = load_digits()
print(digits.keys())
print('data_type:',digits.data.dtype)
print('data_feature_name:',digits.feature_names)
print('data_feature.shape:',digits.data.shape)
df_train = pd.DataFrame(digits.data,columns=digits.feature_names)
df_train['Target'] = digits.target
train_digits = df_train.drop('Target',axis=1)
train_digits_T = df_train['Target']
print(df_train.isnull().sum().values>0)#可見沒有~~
x_train_d,x_test_d,y_train_d,y_test_d = train_test_split(train_digits,train_digits_T,test_size=0.25,random_state=4)
#建立模型
clf_gbc = GradientBoostingClassifier()
#訓練模型
clf_gbc.fit(x_train_d,y_train_d)
#模型預測
y_pred_d = clf_gbc.predict(x_test_d)
acc_d = accuracy_score(y_test_d,y_pred_d)
print('Accuracy_d:',acc_d)
print('clf_gbc.n_estimators:',clf_gbc.n_estimators)
score = []
for i in np.linspace(0.01,0.500,21):
    clf_gbc = GradientBoostingClassifier(learning_rate=i)
    clf_gbc.fit(x_train_d,y_train_d)
    y_pred_d = clf_gbc.predict(x_test_d)
    temp = accuracy_score(y_test_d,y_pred_d)
    score.append(temp)
plt.plot(np.linspace(0.01,0.500,21),score,color = 'r')
plt.xlabel('learning_rate')
plt.ylabel('score')
plt.show()
print('best learning rate:',np.linspace(0.01,0.500,21)[score.index(max(score))])
print('max score:',max(score))