# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:53:48 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO REDES NEURAIS
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:,3:19].values
y = df.iloc[:,2:3].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

regressor = MLPRegressor(hidden_layer_sizes=(9,9))
regressor.fit(X_train, y_train)

score = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

previsoes = regressor.predict(X_test)
y_test = scaler_y.inverse_transform(y_test)
previsoes = scaler_y.inverse_transform(previsoes)

mae = mean_absolute_error(y_test, previsoes)