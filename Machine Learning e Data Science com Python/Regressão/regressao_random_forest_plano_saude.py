# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:49:08 2020

Amostra para predição de custo
Base de dados plano_saude2.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO COM RANDOM FOREST
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('CSVs/plano_saude2.csv')

X = df.iloc[:,0:1].values
y = df.iloc[:, 1].values

regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X, y)
score = regressor.score(X, y)

X_test = np.arange(min(X), max(X), 0.1)
X_test = X_test.reshape(-1,1)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X_test, regressor.predict(X_test), color = 'r')
ax.set_title('Regressão com random forest')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')

previsao = regressor.predict([[40]])
