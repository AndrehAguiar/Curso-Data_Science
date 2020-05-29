# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:58:50 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR SIMPLES
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:, 5:6].values
y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
score = regressor.score(X_train, y_train)

fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
ax.plot(X_train, regressor.predict(X_train), color='r')
ax.set_title('Dataset Treino')
ax.set_xlabel('House Size')
ax.set_ylabel('House Price')
plt.show()

previsoes = regressor.predict(X_test)

resultado = abs(y_test - previsoes)
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.plot(X_test, regressor.predict(X_test), color='r')
ax.set_title('Dataset Teste')
ax.set_xlabel('House Size')
ax.set_ylabel('House Price')
plt.show()

regressor.score(X_test, y_test)