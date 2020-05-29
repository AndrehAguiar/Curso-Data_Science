# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:49:08 2020

Amostra para predição de custo
Base de dados plano_saude2.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO COM REGRESSÃO LINEAR SVR
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CSVs/plano_saude2.csv')

X = df.iloc[:,0:1].values
y = df.iloc[:,1:2].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Kernel Linear
regressor_linear = SVR(kernel='linear')
regressor_linear.fit(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor_linear.predict(X), color = 'r')
ax.set_title('Regressão linear(SVR)')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')

score_linear = regressor_linear.score(X, y)

# Kernel POLY
regressor_poly = SVR(kernel='poly', degree = 3)
regressor_poly.fit(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor_poly.predict(X), color = 'r')
ax.set_title('Regressão linear(POLY)')
ax.set_xlabel('Idade')

score_poly = regressor_poly.score(X, y)

# Kernel rbf

regressor_rbf = SVR(kernel='rbf')
regressor_rbf.fit(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor_rbf.predict(X), color = 'r')
ax.set_title('Regressão linear(RBF)')
ax.set_xlabel('Idade')

score_rbf = regressor_rbf.score(X, y)

nova_entrada = [[40]]
previsao_linear = scaler_y.inverse_transform(
    regressor_linear.predict(scaler_X.transform(nova_entrada)))

previsao_poly = scaler_y.inverse_transform(
    regressor_poly.predict(scaler_X.transform(nova_entrada)))

previsao_rbf = scaler_y.inverse_transform(
    regressor_rbf.predict(scaler_X.transform(nova_entrada)))