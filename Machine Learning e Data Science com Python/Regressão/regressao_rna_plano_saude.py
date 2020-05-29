# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:49:08 2020

Amostra para predição de custo
Base de dados plano_saude2.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO COM REDES NEURAIS
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('CSVs/plano_saude2.csv')

X = df.iloc[:,0:1].values
y = df.iloc[:,1:2].values

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor = MLPRegressor()
regressor.fit(X, y)
score = regressor.score(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor.predict(X), color = 'r')
ax.set_title('Regressão ...')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')

previsao = scaler_y.inverse_transform(
    regressor.predict(scaler_X.transform([[40]])))