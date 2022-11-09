#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:24:26 2020

@author: pouria kourehpaz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


# Synthetic Data Generation

X_all_1=X_all[X_all['Class3']==1]
X_all_2=X_all[X_all['Class3']==2]
X_all_3=X_all[X_all['Class3']==3]


X_f_all=pd.concat([X_all_1,X_all_2, X_all_3],axis=0)

X_f=X_f_all.drop(['SDR','Class3'],axis=1)

Y=(X_f_all['Class3'])


from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1) #corss-validation
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier

alg = DecisionTreeClassifier()

max_features = [7]

max_depth= [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
grid = dict(max_features=max_features, max_depth=max_depth)
grid_search = GridSearchCV(estimator=alg, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1_macro',error_score=0, return_train_score=True)
grid_result = grid_search.fit(X_f, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means_test = grid_result.cv_results_['mean_test_score']
means_train = grid_result.cv_results_['mean_train_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means_test, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
split0_test=grid_result.cv_results_['split0_test_score']; split0_train=grid_result.cv_results_['split0_train_score'] 
split1_test=grid_result.cv_results_['split1_test_score']; split1_train=grid_result.cv_results_['split1_train_score'] 
split2_test=grid_result.cv_results_['split2_test_score']; split2_train=grid_result.cv_results_['split2_train_score'] 
split3_test=grid_result.cv_results_['split3_test_score']; split3_train=grid_result.cv_results_['split3_train_score'] 
split4_test=grid_result.cv_results_['split4_test_score']; split4_train=grid_result.cv_results_['split4_train_score'] 
split5_test=grid_result.cv_results_['split5_test_score']; split5_train=grid_result.cv_results_['split5_train_score'] 
split6_test=grid_result.cv_results_['split6_test_score']; split6_train=grid_result.cv_results_['split6_train_score'] 
split7_test=grid_result.cv_results_['split7_test_score']; split7_train=grid_result.cv_results_['split7_train_score'] 
split8_test=grid_result.cv_results_['split8_test_score']; split8_train=grid_result.cv_results_['split8_train_score'] 
split9_test=grid_result.cv_results_['split9_test_score']; split9_train=grid_result.cv_results_['split9_train_score'] 
split10_test=grid_result.cv_results_['split10_test_score']; split10_train=grid_result.cv_results_['split10_train_score'] 
split11_test=grid_result.cv_results_['split11_test_score']; split11_train=grid_result.cv_results_['split11_train_score'] 
split12_test=grid_result.cv_results_['split12_test_score']; split12_train=grid_result.cv_results_['split12_train_score'] 
split13_test=grid_result.cv_results_['split13_test_score']; split13_train=grid_result.cv_results_['split13_train_score'] 
split14_test=grid_result.cv_results_['split14_test_score']; split14_train=grid_result.cv_results_['split14_train_score'] 


lower_error_train = means_train-np.array([split0_train,split1_train,split2_train,split3_train,split4_train,split5_train,split6_train,split7_train,split8_train,split9_train,split10_train,split11_train,split12_train,split13_train,split14_train]).min(axis=0)
upper_error_train = -means_train+np.array([split0_train,split1_train,split2_train,split3_train,split4_train,split5_train,split6_train,split7_train,split8_train,split9_train,split10_train,split11_train,split12_train,split13_train,split14_train]).max(axis=0)
asymmetric_error_train = [lower_error_train, upper_error_train]

plt.figure(figsize=(3.25,2), dpi=300)
p1=plt.errorbar(max_depth, means_train, color='k', yerr=asymmetric_error_train, linewidth=1.0, capsize=1, capthick=1.5)


lower_error_test = means_test-np.array([split0_test,split1_test,split2_test,split3_test,split4_test,split5_test,split6_test,split7_test,split8_test,split9_test,split10_test,split11_test,split12_test,split13_test,split14_test]).min(axis=0)
upper_error_test = -means_test+np.array([split0_test,split1_test,split2_test,split3_test,split4_test,split5_test,split6_test,split7_test,split8_test,split9_test,split10_test,split11_test,split12_test,split13_test,split14_test]).max(axis=0)
asymmetric_error_test = [lower_error_test, upper_error_test]

p2=plt.errorbar(max_depth, means_test, color=(0.78,0,0), yerr=asymmetric_error_test, linestyle='dashed', linewidth=1.0, capsize=1, capthick=1.5)

error_matrix=np.transpose(np.array([lower_error_test, upper_error_test, lower_error_train, upper_error_train]))      
    

plt.xlim(4, np.max(max_depth))
plt.ylim(0.7, 1)               
plt.xlabel('Max Tree Depth', fontsize=9)
plt.ylabel('F1-score', fontsize=9)
#plt.xticks(np.arange(0, 5.5, step=1), fontsize=9)
plt.yticks(fontsize=9)

plt.legend([p1[0],p2[0]],['train','test'], fontsize=9)
plt.title('Decision Tree', fontsize=9)
plt.show()


