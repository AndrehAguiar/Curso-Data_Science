# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:53:48 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR POLINOMIAL
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:,3:19].values
y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

score = regressor.score(X_train_poly, y_train)

previsoes = regressor.predict(X_test_poly)

mae = mean_absolute_error(y_test, previsoes)

regressor.score(X_test_poly, y_test)