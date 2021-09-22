import os
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

dir = './Python_Practice/Python_Pandas/Datasets/'
path = os.path.join(dir,'titanic_train.csv')
df = pd.read_csv(path)

train_T = df['Survived']
df = df.drop(['Survived','PassengerId'],axis=1)
print(df.head())
#特徵工程
LabE = LabelEncoder()
MME = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LabE.fit_transform(list(df[c].values))
    df[c] = MME.fit_transform(df[c].values.reshape(-1,1))
print(df.head())

train = df.values
# 因為邏輯斯迴歸時也要資料，因此將訓練集切成三個部分 train / val / test ，採用test 驗證而非k-fold交叉驗證
# train 用來訓練梯度提升樹, val 用來訓練邏輯斯迴歸, test 驗證效果
train_x,test_x,train_y,test_y = train_test_split(train,train_T,test_size=0.5)
train_x, val_x, train_y, val_y = train_test_split(train, train_T, test_size=0.5)
# 隨機森林擬合後, 再將葉編碼 (*.apply) 結果做獨熱 / 邏輯斯迴歸
rf = RandomForestClassifier(n_estimators=20, min_samples_split=10, min_samples_leaf=5, 
                            max_features=4, max_depth=3, bootstrap=True)
onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

rf.fit(train_x,train_y)
onehot.fit(rf.apply(train_x))
lr.fit(onehot.transform(rf.apply(val_x)),val_y)
#將隨機森林+業編碼+邏輯斯回歸結果輸出
pred_rf_lr = lr.predict_proba(onehot.transform(rf.apply(test_x)))[:,1]
fpr_rf_lr,tpr_rf_lr,_ = roc_curve(test_y,pred_rf_lr)
#將隨機森林結果輸出
pred_rf = rf.predict_proba(test_x)[:,1]
fpr_rf,tpr_rf,_ = roc_curve(test_y,pred_rf)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rf,tpr_rf,label='RF')
plt.plot(fpr_rf_lr,tpr_rf_lr,label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()