# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:12:29 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CROSS VALIDATION
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:,3:19].values
y = df.iloc[:,2:3].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

resultados_mean = []
matriz_results = []
# Pré-processamento para regressão linear polinomial
# poly = PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)

for i in range(0,30):
    kfold = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=i)
    
    resultados = []
    
    for indice_treinamento, indice_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
        
        # regressor = LinearRegression()
        # regressor = RandomForestRegressor(n_estimators=100)
        # regressor = DecisionTreeRegressor(criterion='mae')
        # regressor = SVR(kernel='rbf', C=3)
        regressor = MLPRegressor(hidden_layer_sizes=(9,9))
        regressor.fit(X[indice_treinamento], y[indice_treinamento].ravel())
        score = regressor.score(X[indice_teste], y[indice_teste].ravel())
    
    
    resultados.append(score)
    result = np.asarray(resultados)
    resultados_mean.append(result.mean())
    
matriz_results.append(resultados_mean)
results_array = np.asarray(matriz_results)
