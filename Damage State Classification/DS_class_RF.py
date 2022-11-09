#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:06:12 2022

@author: pouriak
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = "Arial"

data=pd.ExcelFile('Input_Data.xlsx')
data_bld=pd.read_excel(data,'Sheet1')
data_gm=pd.read_excel(data,'data_all')

txt='H14SEAWB'
storyid=['S12','S16','S20','S24','S4','S8']
strategy=['PG2','PG3','PG4','PG5','PG6','PG7']
A=np.zeros((len(data_gm),3))
for i in storyid:
    for j in strategy:
        indx_bld=np.where(data_bld['Arch']==i+txt+j)
        indx_gm=np.where(data_gm['Arch']==i+txt+j)   
        A[indx_gm,0]=data_bld['T1'][float(np.asarray(indx_bld))]
        A[indx_gm,1]=data_bld['Ix'][float(np.asarray(indx_bld))]
        A[indx_gm,2]=data_bld['long_reinf_ratio'][float(np.asarray(indx_bld))]
B=pd.DataFrame(A)
X1=pd.concat([B,data_gm['SA_avg']],axis=1)
X2=pd.concat([X1,data_gm['Ds']],axis=1)
X3=pd.concat([X2,data_gm['SI']],axis=1)
X4=pd.concat([X3,data_gm['DSI']],axis=1)
X5=pd.concat([X4,data_gm['GM_ID']],axis=1)

X=pd.concat([X5,data_gm['Class3']],axis=1)
X_all=pd.concat([X,data_gm['SDR']],axis=1)

X_all_1=X_all[X_all['Class3']==1]
X_all_2=X_all[X_all['Class3']==2]
X_all_3=X_all[X_all['Class3']==3]


X_f_all=pd.concat([X_all_1,X_all_2, X_all_3],axis=0)

X_f=X_f_all.drop(['SDR','Class3'],axis=1)

Y=(X_f_all['Class3'])



from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Implement RF
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_f,Y,test_size = 0.2, stratify=Y)


filename = 'RF_final.sav'
alg = pickle.load(open(filename, 'rb'))

Y_pred = alg.predict(X_test)
Y_pred_train = alg.predict(X_train)

M_train = confusion_matrix(Y_train, Y_pred_train) 
M_test = confusion_matrix(Y_test, Y_pred) 

a1=M_test[0,0]/np.sum(M_test[0,:])
a2=M_test[1,1]/np.sum(M_test[1,:])
a3=M_test[2,2]/np.sum(M_test[2,:])

a_train = accuracy_score(Y_pred_train, Y_train)
a_test = accuracy_score(Y_pred, Y_test)

f1_train = f1_score(Y_train, Y_pred_train, average='macro')
f1_test = f1_score(Y_test, Y_pred, average='macro')
