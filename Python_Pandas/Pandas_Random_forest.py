from sklearn.datasets import load_iris,load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from itertools import compress
from sklearn.linear_model import Lasso
#讀取iris
iris = load_iris()
#資料切分
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=4)
#建立模型
clf = RandomForestClassifier()
#訓練模型
clf.fit(x_train,y_train)
#模型預測
y_pred = clf.predict(x_test)
print('Accuracy:{}'.format(accuracy_score(y_test,y_pred)))
print('iris keys:',iris.keys())
print('iris index:',iris.feature_names)
print('Feature importance:',clf.feature_importances_)
#遊玩兼練習系列
wine = load_wine()
#資料描述
print('wine_key:',wine.keys())
print('wine_features:',wine.feature_names)
print('wine dtype:',wine.data.dtype)
print('wine target_names:',wine.target_names)
wine_df = pd.DataFrame(wine.data,columns=wine.feature_names)
wine_df['target'] = wine.target
train_T = wine_df['target']
train_wine = wine_df.drop(['target'],axis=1)
print(train_wine.isnull().sum())
MME = MinMaxScaler()
#相關係數
correlations = wine_df.corr()['target'].sort_values()
print('Most Positive Correlations:\n{}'.format(correlations.tail(4)))
print('Most Negative Correlations:\n{}'.format(correlations.head(6)))
feature_mask_c = list((correlations>0.4)|(correlations<-0.6)&(correlations!=1))
feature_list_c = list(compress(list(train_wine),feature_mask_c))
print(feature_list_c)
#使用隨機森林選取特徵
est_RF = RandomForestClassifier()
train_wine_RF = MME.fit_transform(train_wine)
est_RF.fit(train_wine,train_T)
feats = pd.Series(data = est_RF.feature_importances_,index=train_wine.columns)
feats = feats.sort_values(ascending=False)
print(feats)
#使用L1 Embedding 做特徵選擇(自訂門檻)
est_L = Lasso(alpha=0.005)
train_L1 = MME.fit_transform(train_wine)
est_L.fit(train_L1,train_T)
print('L1_Reg:{}'.format(est_L.coef_))
#做一點處理
L1_mask = list((est_L.coef_>0.5)|(est_L.coef_<-1))
L1_list = list(compress(list(train_wine),list(L1_mask)))
print('L1_list:{}'.format(L1_list))
#簡易相關度提取最佳
train_c = MME.fit_transform(train_wine[feature_list_c])
x_train_c,x_test_c,y_train_c,y_test_c = train_test_split(train_c,train_T,test_size=0.25,random_state=4)
#隨機森林選取特徵相最佳
feats_RF = list(feats[:5].index)
train_RF = MME.fit_transform(train_wine[feats_RF])
x_train_RF,x_test_RF,y_train_RF,y_test_RF = train_test_split(train_RF,train_T,test_size=0.25,random_state=4)
#選取L1最佳特徵
train_L1 = MME.fit_transform(train_wine[L1_list])
x_train_L1,x_test_L1,y_train_L1,y_test_L1 = train_test_split(train_L1,train_T,test_size=0.25,random_state=4)
#建立模型
clf = RandomForestClassifier()
#訓練模型使用簡易相關係數
clf.fit(x_train_c,y_train_c)
#模型預測簡易相關係數
y_pred_c = clf.predict(x_test_c)
acc = accuracy_score(y_test_c,y_pred_c)
print("Acuuracy_RF_c: ", acc)
#訓練模型隨機森林選取特徵
clf.fit(x_train_RF,y_train_RF)
#模型預測簡易相關係數
y_pred_RF = clf.predict(x_test_RF)
acc = accuracy_score(y_test_RF,y_pred_RF)
print("Acuuracy_RF_RF: ", acc)
#訓練模型隨機森林選取特徵
clf.fit(x_train_L1,y_train_L1)
#模型預測簡易相關係數
y_pred_L1 = clf.predict(x_test_L1)
acc = accuracy_score(y_test_L1,y_pred_L1)
print("Acuuracy_RF_L1: ", acc)