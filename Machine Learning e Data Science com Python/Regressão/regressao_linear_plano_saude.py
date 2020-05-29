# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:29:10 2020

Amostra para predição de custo
Base de dados plano_saude.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR SIMPLES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

df = pd.read_csv('CSVs/plano_saude.csv')

# Separa preditores(X) e targets(y)
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# verifica a correlação entre as variáveis
correlacao = np.corrcoef(X, y)

# Transforma vetor em matriz
X = X.reshape(-1,1)

# Instancia a classe do modelo
regressor = LinearRegression()

# Treina o modelo
regressor.fit(X, y)

# b1
const = regressor.intercept_
# b0
coef = regressor.coef_

fig, ax = plt.subplots()
ax.set_title(f'Regressão Linear Simples\nb0={round(coef[0],4)} / b1={round(const, 4)}')
ax.scatter(X, y, color='r')
ax.plot(X, regressor.predict(X))
ax.set_xlabel('Idade')
ax.set_ylabel('Valor')
ax.grid(True)
plt.show()

previsao1 = regressor.predict([[40]])
previsao2 = const + coef * 40

score = regressor.score(X, y)

visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()