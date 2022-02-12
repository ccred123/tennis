# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:52:45 2021

@author: User
"""

import seaborn as sns
import pandas as pd
import statsmodels.api as sm
#import numpy as np
from sklearn.model_selection import StratifiedKFold
from statsmodels.discrete.discrete_model import Probit
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
import numpy as np


#讀取檔案
#excelpath='修frenchopen.csv'
data=pd.read_csv ('修frenchopen20192020.csv')
#print(data)
#print(data.head())
#print(data.describe())
Y = data["result"]
X = data.drop(["result"],1)
#加常數項(截距)
X =sm.add_constant(X)

#畫x變數及y因變數的圖
#x = data['p21spw'].values
#y = data['result'].values
#plt.scatter(x, y)
#plt.ylabel('result')
#plt.xlabel('p1spw')
#plt.show()


model = Probit(Y,X)
#擬合模型
probit_model = model.fit()
print(probit_model.summary())
#轉換為預測的值
probit_predictY=probit_model.predict(X)
print(probit_predictY)
probit_predictY.plot()
#把預測的值轉成整數
probit_int_predictY=probit_predictY.astype(int)
print(probit_int_predictY)


#每個因變數的cdf
model.cdf(X.iloc[1,:])
#算出來的係數乘第一項的x值
estY=sum(probit_model.params*X.loc[1])
print("cdf=",estY)
model.cdf(estY)
#probit的精準度
print('Test accuracy = ', accuracy_score(Y, probit_int_predictY))


#PROBIT預測下一場比賽
Y2 = data["result"]
X2 = data.drop(["result"],1)
model2 = Probit(Y2,X2.astype(float)).fit()
nextgame=model2.predict([0.81,0.63,0.56,0.71,23,0.78,0.57,0.45,0.58,43])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.84,0.58,0.47,0.68,23,0.7,0.52,0.27,0.56,37])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.79,0.62,0.55,0.87,29,0.56,0.59,0.45,0.5,21])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.81,0.58,0.43,0.92,30,0.42,0.66,0.38,0.47,40])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.73,0.66,0.68,0.73,29,0.59,0.63,0.46,0.83,39])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.796,0.614,0.538,0.782,26.8,0.802,0.676,0.564,0.704,24.4])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
nextgame=model2.predict([0.67,0.65,0.66,0.53,14,0.5,0.67,0.48,0.67,52])
result=nextgame.astype(int)
print('下一場比賽結果:',result)
#==============================================================================
#probit混淆矩陣
def plot_confusion_matrix1(cm, classes,
                          normalize=True,
                          title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, 
               interpolation='nearest', 
               cmap='Blues')#interpolation:紋路，cmap:顏色
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="w" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
target_names = [ 'p1win(0)','p1loss(1)']
w= confusion_matrix(Y, probit_int_predictY) 
plot_confusion_matrix1(w, 
                       classes=target_names,
                       normalize=False,
                       title=' confusion matrix')
plt.show()
print("========================特徵選取=========================================")
#method 2(特徵選取)
method_reg = LogisticRegression()#LinearRegression()
#選擇K個特徵(為0LS選出,r-squared:0.8)
#selector_RFE = RFE(method_reg,5, step=1)

#rfecv訓練的模型
ref_model=RFECV(estimator=method_reg, step=1,scoring='accuracy').fit(X,Y )
print("Optimal number of features : %d" % ref_model.n_features_)
print("Selected Features: %s" % (ref_model.support_))
print("Feature Ranking: %s" % (ref_model.ranking_))
lrfeXk=X[X.columns[ref_model.support_]]

#變數轉換為預測值
lpredictY=ref_model.predict(X)
print(lrfeXk)
#將預測值轉為整數
lintpredY=lpredictY.astype(int)
print(lintpredY)
#rfecv精準度
print('Test accuracy = ', accuracy_score(Y, lintpredY))

#RFECV的圖
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(ref_model.grid_scores_)+1), ref_model.grid_scores_)
plt.show()
#==============================================================================
#RFE的混淆矩陣
def plot_confusion_matrix2(cm, classes,
                          normalize=False,#百分比
                          title='RFE_Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)                 
    plt.imshow(cm, 
               interpolation='nearest', 
               cmap='Blues')#interpolation:漸層，cmap:顏色
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="w" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
target_names = [ 'p1win(0)','p1loss(1)']
w= confusion_matrix(Y, lintpredY)
plot_confusion_matrix2(w, 
                       classes=target_names,
                       normalize=False,
                       title=' RFE_confusion matrix')
plt.show()
#==============================================================================
#rfecv的模型預測下一場比賽
Y3 = data["result"]
X3 = data.drop(["result"],1)
ref_model2=RFECV(estimator=method_reg, step=1,scoring='accuracy').fit(X3,Y3 )

# 建立list
#to_be_predicted = []  
#x3=[p11spw,p11sp,p12spw,p1net,p1error,p21spw,p21spw,p22spw,p2net,p2error]
#for i in range(0, 9):
#    for j in range(0,9):
#        ele = print(":",x3.iloc[j],int(input()))
#        to_be_predicted.append(ele) # adding the element
    
to_be_predicted = np.array([
   [0.81,0.63,0.56,0.71,23,0.78,0.57,0.45,0.58,43],
   [0.84,0.58,0.47,0.68,23,0.7,0.52,0.27,0.56,37],
   [0.79,0.62,0.55,0.87,29,0.56,0.59,0.45,0.5,21],
   [0.81,0.58,0.43,0.92,30,0.42,0.66,0.38,0.47,40],
   [0.73,0.66,0.68,0.73,29,0.59,0.63,0.46,0.83,39],
   [0.796,0.614,0.538,0.782,26.8,0.802,0.676,0.564,0.704,24.4],
   [0.67,0.65,0.66,0.53,14,0.5,0.67,0.48,0.67,52]
   
])
nextgame2=ref_model2.predict(to_be_predicted)
result2=nextgame2.astype(int)
print('下一場比賽結果:',result2)