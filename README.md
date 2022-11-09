# ML-for-Building-Damage-Classification
<p align="center">
<img src="https://github.com/pkourehpaz/ML-for-Building-Damage-Classification/blob/main/ML_framework.jpg" 
	height="350"/>
</p>
	    
</p>

## Introduction

This repository presents a machine learning-based framework to predict a building’s post-earthquake damage state using structural properties and ground motion intensity measures as model inputs. This tool implements the framework presented in: 

Kourehpaz, P. and Molina Hutt, C. (2022). “Machine Learning for Enhanced Regional Seismic Risk Assessments” *Journal of Structural Engineering*, 148(9): 04022126. https://ascelibrary.org/doi/10.1061/%28ASCE%29ST.1943-541X.0003421

## Modules

This repository consists of three different modules as follows: 

- **Hyperparameter Optimization** 
- **Damage State Classification** 
- **Collapse Status Identification**

## Basic Demo

To determine a building's damage state: 

```python
X_input = np.transpose(pd.DataFrame(np.array(['T1','I_x','rho_x','SA_avg','Ds','SI','DSI','GM_ID']))) #insert input parameters; note that GM_ID =0 for emprerical GMs & GM_ID = 1 for simulated GMs

model = 'GBoost_final.sav' # the available models are logistic regression (LR), k-nearest neighbors (KNN), decision tree, random forest (RF), Adaboost, and gradient boosting (GBoost)
loaded_alg = pickle.load(open(model, 'rb'))

Y_pred = loaded_alg.predict(X_input) #predicted damage state (1: negligible damage; 2: minor-moderate damage; 3: severe damage)
```

To determine a building's collapse status: 

```python
X_input = np.transpose(pd.DataFrame(np.array(['T1','I_x','rho_x','SA_avg','Ds','SI','DSI','GM_ID']))) #insert input parameters; note that GM_ID =0 for emprerical GMs & GM_ID = 1 for simulated GMs

model = 'GBoost_col_syn.sav' #collapse status identification using synthetic data
loaded_alg = pickle.load(open(model, 'rb'))

Y_pred = loaded_alg.predict(X_input) #predicted collapse status (0: no collapse; 1: collapse)
```
