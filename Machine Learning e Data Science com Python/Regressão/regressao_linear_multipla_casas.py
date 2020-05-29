# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:58:50 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR SIMPLES
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:, 3:19].values
y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
score = regressor.score(X_train, y_train)
regressor.score(X_test, y_test)
previsoes = regressor.predict(X_test)

# Média de erro absoluto
mae = mean_absolute_error(y_test, previsoes)

# Média de erro quadratico
mse = mean_squared_error(y_test, previsoes)

# b0
coef = regressor.coef_

# b1
const = regressor.intercept_
regressor.score(X_test, y_test)