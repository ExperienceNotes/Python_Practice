
from sklearn import preprocessing
from sklearn.datasets import load_boston,load_iris,load_wine
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress
import warnings
warnings.filterwarnings('ignore')
#load_data
iris = load_iris()
#!通常這邊會做前處理接著做特徵工程，但現在只是練習樹模型
#資料切割
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=4)
#建立模型
clf = DecisionTreeClassifier()
#訓練模型
clf.fit(x_train,y_train)
#預測測試集
pred_y = clf.predict(x_test)
acc = accuracy_score(y_test,pred_y)
print("Acuuracy: ", acc)
#print(iris的特徵)
print(iris.feature_names)
#print(iris的重要特徵)
print("Feature importance: ", clf.feature_importances_)
#--------------------------------------------------------
#開始玩其他Data
boston = load_boston()
wine = load_wine()
#資料描述
print('boston.keys:',boston.keys())
print('boston dtype:',boston.data.dtype)
print('boston.data.shape:',boston.data.shape)
print('boston.data.columns',boston.feature_names)
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
#print(boston_df.head())
boston_df['MEDV'] = boston.target
boston_train = boston_df.drop(['MEDV'],axis=1)
boston_T = boston_df['MEDV']
#Data前處理
#檢查是否有空值
print(boston_df.isnull().sum())
#EDA(資料探索)
new_feature = []
for dtype,feature in zip(boston_df.dtypes,boston_df.columns):
    if dtype == 'int64' or dtype == 'float64':
        new_feature.append(feature)
print("new_feature:{}".format(new_feature))
'''
for col in new_feature:
    fig = plt.figure(figsize=(18,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    boston_df.boxplot(column = col, ax = ax1)
    boston_df.hist(column = col, ax = ax2)
    plt.show()
'''
MME = MinMaxScaler()
#簡易取得相關度
correlations = boston_df.corr()['MEDV'].sort_values()
print('Most Positive Correlations:\n{}'.format(correlations.tail(3)))
print('Most Negative Correlations:\n{}'.format(correlations.head(3)))
#使用隨機森林選取特徵
est_R_F = RandomForestRegressor()
est_R_F.fit(boston_train.values,boston_T)
feats = pd.Series(data = est_R_F.feature_importances_,index=boston_train.columns)
feats = feats.sort_values(ascending=False)
print(feats)
#使用L1 Embedding 做特徵選擇(自訂門檻)
est_L = Lasso(alpha=0.01)
train_L1 = MME.fit_transform(boston_train)
est_L.fit(train_L1,boston_T)
print('L1_Reg:{}'.format(est_L.coef_))
#做一點處理
L1_mask = list((est_L.coef_>2)|(est_L.coef_<-10))
L1_list = list(compress(list(boston_train),list(L1_mask)))
print('L1_list:{}'.format(L1_list))
#簡易相關度提取最佳相關度RM、ZN、LSTAT、PTRATIO、INDUS
train_corr = boston_train[['RM','ZN','LSTAT','PTRATIO','INDUS']]
train_corr = MME.fit_transform(train_corr)
#提取隨機森林做出來的feat
feats_R = list(feats[:2].index)
train_R = boston_train[feats_R]
train_R = MME.fit_transform(train_R)
#提取L1 特徵門檻
train_L1_feats = boston_train[L1_list]
train_L1_feats = MME.fit_transform(train_L1_feats)
#分割資料
#相關係數
x_train_c,x_test_c,y_train_c,y_test_c = train_test_split(train_corr,boston_T,test_size=0.25,random_state=4)
#隨機森林
x_train_RF,x_test_RF,y_train_RF,y_test_RF = train_test_split(train_R,boston_T,test_size=0.25,random_state=4)
#L1 套索回歸
x_train_L1,x_test_L1,y_train_L1,y_test_L1 = train_test_split(train_L1_feats,boston_T,test_size=0.25,random_state=4)
#建立模型使用Random F Note:順便玩一下參數
clf_p = DecisionTreeRegressor(criterion='mse',
                              min_samples_split= 12,
                              min_samples_leaf= 4
                              )
#訓練模型使用corr
clf_p.fit(x_train_c,y_train_c)
#模型預測corr
y_pred_c = clf_p.predict(x_test_c)
print("Mean squared error_DTR_C: %.2f"
      % mean_squared_error(y_test_c, y_pred_c))
#訓練模型使用隨機森林的特徵
clf_p.fit(x_train_RF,y_train_RF)
#模型預測RandomF
y_pred_RF = clf_p.predict(x_test_RF)
print("Mean squared error_DTR_RF: %.2f"
      % mean_squared_error(y_test_RF, y_pred_RF))
#訓練模型使用L1 Lasso的特徵
clf_p.fit(x_train_L1,y_train_L1)
y_pred_L1 = clf_p.predict(x_test_L1)
print("Mean squared error_DTR_L1: %.2f"
      % mean_squared_error(y_test_L1, y_pred_L1))