# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:11:31 2020

Amostra para predição de custo
Base de dados plano_saude2.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA ÁRVORE DE REGRESSÃO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('CSVs/plano_saude2.csv')

X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

regressor = DecisionTreeRegressor()
regressor.fit(X, y)
score = regressor.score(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor.predict(X), color = 'r')
ax.set_title('Regressão com árvores')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')
plt.show()

X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1,1)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X_teste, regressor.predict(X_teste), color = 'r')
ax.set_title('Regressão com árvores')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')
plt.show()

regressor.predict([[40]])