#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:47:51 2022

@author: pouriak
"""
import numpy as np
import pandas as pd
import pickle

X_input = np.transpose(pd.DataFrame(np.array(['T1','I_x','rho_x','SA_avg','Ds','SI','DSI','GM_ID']))) #insert input parameters; note that GM_ID =0 for emprerical GMs & GM_ID = 1 for simulated GMs

model = 'GBoost_final.sav'
loaded_alg = pickle.load(open(model, 'rb'))

Y_pred = loaded_alg.predict(X_input) #predicted damage state (1: negligible damage; 2: minor-moderate damage; 3: severe damage)

