#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:24:26 2020

@author: pouriak
"""
import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import SVMSMOTE

data=pd.ExcelFile('Input_Data.xlsx')
data_bld=pd.read_excel(data,'Sheet1')
data_gm=pd.read_excel(data,'Sheet2')

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

X_all_1r=X_all_1.sample(n=4261)
X_all_2r=X_all_2.sample(n=4261)
X_all_3r=X_all_3.sample(n=4261)


##balanced data
X_f_all=pd.concat([X_all_1r, X_all_2r, X_all_3r],axis=0)


#sdr = [0.039, 0.0427, 0.045, 0.0485, 0.053] #other SDR thresholds may be chosen by a user
sdr=0.045
#collapse damage state
X_f_all=X_f_all[X_f_all['SDR']>=0.0]
Y = np.zeros(len(X_f_all))
for i in range(len(Y)):
    if X_f_all['SDR'].iloc[i]>=sdr:
        Y[i]=1
Y = pd.Series(Y)

X_f=X_f_all.drop(['SDR','Class3'],axis=1)

sm=SVMSMOTE(random_state=42)
X_b1, Y_b1 = sm.fit_resample(X_f, Y) #balanced data



# merge PSHA with M9 data
data=pd.ExcelFile('Input_Data.xlsx')
data_bld=pd.read_excel(data,'Sheet1')
data_gm=pd.read_excel(data,'Sheet3')

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


X_f=X_all.drop(['SDR','Class3'],axis=1)
Y=(X_all['Class3'])

X_all_1=X_all[X_all['Class3']==1]
X_all_2=X_all[X_all['Class3']==2]
X_all_3=X_all[X_all['Class3']==3]

X_all_1r=X_all_1.sample(n=200)
X_all_2r=X_all_2.sample(n=200)
X_all_3r=X_all_3.sample(n=180)

X_f_all=pd.concat([X_all_1r,X_all_2r, X_all_3r],axis=0)


#collapse damage state
X_f_all=X_f_all[X_f_all['SDR']>=0.0]
Y = np.zeros(len(X_f_all))
for i in range(len(Y)):
    if X_f_all['SDR'].iloc[i]>=sdr:
        Y[i]=1
Y = pd.Series(Y)

X_f=X_f_all.drop(['SDR','Class3'],axis=1)

sm=SVMSMOTE(random_state=42)
X_b2, Y_b2 = sm.fit_resample(X_f, Y) #balanced data




from sklearn.model_selection import train_test_split
X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X_b1,Y_b1,test_size = 0.2, stratify=Y_b1)
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_b2,Y_b2,test_size = 0.2, stratify=Y_b2)

X_train = pd.concat([X_train1,X_train2])
X_test = pd.concat([X_test1,X_test2])
Y_train = pd.concat([Y_train1,Y_train2])
Y_test = pd.concat([Y_test1,Y_test2])


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from sklearn.ensemble import GradientBoostingClassifier
alg = GradientBoostingClassifier(n_estimators=60, learning_rate=0.2, max_depth=4)

alg.fit(X_train,Y_train)
filename = 'Gboost_col_syn.sav'
pickle.dump(alg, open(filename, 'wb'))

Y_pred = alg.predict(X_test)
Y_pred_train = alg.predict(X_train)

M_train = confusion_matrix(Y_train, Y_pred_train) 

M_test = confusion_matrix(Y_test, Y_pred)

a_train = accuracy_score(Y_pred_train, Y_train)
a_test = accuracy_score(Y_pred, Y_test)

a1=M_test[0,0]/np.sum(M_test[0,:])
a2=M_test[1,1]/np.sum(M_test[1,:])


f1_train = f1_score(Y_train, Y_pred_train, average='macro')
f1_test = f1_score(Y_test, Y_pred, average='macro')
f1_test_mean = f1_score(Y_test, Y_pred, average='weighted')

#print(M_test)
#print(f1_test)


#%%
#% Collapse status identification
import numpy as np
import pandas as pd

data=pd.ExcelFile('Input_Data.xlsx')
data_bld=pd.read_excel(data,'Sheet1')
data_gm=pd.read_excel(data,'data_all')

txt='H14SEAWB'
storyid=['S12','S16','S20','S24','S4','S8']
strategy=['','PG2','PG3','PG4','PG5','PG6','PG7']
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


#collapse damage state
X_f_all=X_f_all[X_f_all['SDR']>=0.0]
Y = np.zeros(len(X_f_all))
for i in range(len(Y)):
    if X_f_all['SDR'].iloc[i]>=sdr:
        Y[i]=1
Y = pd.Series(Y)

X_f=X_f_all.drop(['SDR','Class3'],axis=1)


filename = 'Gboost_col_syn.sav'
loaded_alg = pickle.load(open(filename, 'rb'))

Y_pred = loaded_alg.predict(X_f)


M_confusion = confusion_matrix(Y, Y_pred) 

accuracy = accuracy_score(Y_pred, Y)
a1=M_confusion[0,0]/np.sum(M_confusion[0,:])
a2=M_confusion[1,1]/np.sum(M_confusion[1,:])

recall=M_confusion[1,1]/np.sum(M_confusion[1,:])
precision=M_confusion[1,1]/np.sum(M_confusion[:,1])

f1 = f1_score(Y, Y_pred, average='macro')
#print(M_confusion)
print("F1-score is", f1)
print("recall is", recall)
print("precision is", precision)


#%% building portfolio damage state prediction

data=pd.ExcelFile('Regional_Data.xlsx')
data_bld=pd.read_excel(data,'Sheet1')
data_gm=pd.read_excel(data,'Seattle') #insert site location


#% machine learning prediction
txt='H14SEAWB'
storyid=['S12','S16','S20','S24','S4', 'S8']
strategy=['','PG2','PG3','PG4','PG5','PG6','PG7']
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

X=pd.concat([X5,data_gm['Class_col']],axis=1)
X_all=pd.concat([X,data_gm['SDR']],axis=1)


X_f=X_all.drop(['SDR','Class_col'],axis=1)

Y=(X_all['Class_col'])


# prediction
filename = 'GBoost_col_syn.sav'
loaded_alg = pickle.load(open(filename, 'rb'))

Y_pred = loaded_alg.predict(X_f)


M_confusion = confusion_matrix(Y, Y_pred) 

accuracy = accuracy_score(Y_pred, Y)

f1 = f1_score(Y, Y_pred, average='macro')
print(M_confusion)
print(f1)



