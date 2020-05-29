# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:53:48 2020

Amostra para predição de custo
Base de dados house_prices.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR POLINOMIAL
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

df = pd.read_csv('CSVs/house_prices.csv')

X = df.iloc[:,3:19].values
y = df.iloc[:, 2:3].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Kernel Linear
regressor_linear = SVR(kernel='linear')
regressor_linear.fit(X_train, y_train.ravel())
score_linear = regressor_linear.score(X_train, y_train.ravel())
score_linear_test = regressor_linear.score(X_test, y_test.ravel())

previsoes_linear = regressor_linear.predict(X_test)

# Kernel POLY
regressor_poly = SVR(kernel='poly', degree=3)
regressor_poly.fit(X_train, y_train.ravel())
score_poly = regressor_poly.score(X_train, y_train.ravel())
score_test_poly = regressor_poly.score(X_test, y_test.ravel())
previsoes_poly = regressor_poly.predict(X_test)

# Kernel POLY
regressor_rbf = SVR(kernel='rbf', C=3)
regressor_rbf.fit(X_train, y_train.ravel())
score_rbf = regressor_rbf.score(X_train, y_train.ravel())
score_test_rbf = regressor_rbf.score(X_test, y_test.ravel())
previsoes_rbf = regressor_rbf.predict(X_test)

y_test = scaler_y.inverse_transform(y_test)

previsoes_linear = scaler_y.inverse_transform(previsoes_linear)
previsoes_poly = scaler_y.inverse_transform(previsoes_poly)
previsoes_rbf = scaler_y.inverse_transform(previsoes_rbf)

mae_linear = mean_absolute_error(y_test, previsoes_linear)
mae_poly = mean_absolute_error(y_test, previsoes_poly)
mae_rbf = mean_absolute_error(y_test, previsoes_rbf)