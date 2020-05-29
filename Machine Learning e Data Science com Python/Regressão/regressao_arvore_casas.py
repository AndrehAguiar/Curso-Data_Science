# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:25:19 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA ÁRVORE DE REGRESSÃO
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:,3:19].values
y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

score = regressor.score(X_train, y_train)
previsoes = regressor.predict(X_test)

mae = mean_absolute_error(y_test, previsoes)

regressor.score(X_test, y_test)