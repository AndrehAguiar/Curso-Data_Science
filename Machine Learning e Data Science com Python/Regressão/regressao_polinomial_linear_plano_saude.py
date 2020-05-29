# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:53:48 2020

Amostra para predição de custo
Base de dados plano_saude2.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA REGRESSÃO LINEAR POLINOMIAL
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('CSVs/plano_saude2.csv')

X = df.iloc[:,0:1].values
y = df.iloc[:, 1].values

regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

regressor1.predict([[40]])

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor1.predict(X), color='r')
ax.set_title('Regressão linear')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')
plt.show()

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
score2 = regressor2.score(X_poly, y)

regressor2.predict(poly.fit_transform([[40]]))

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, regressor2.predict(poly.fit_transform(X)), color='r')
ax.set_title('Regressão polinomial')
ax.set_xlabel('Idade')
ax.set_ylabel('Custo')
plt.show()