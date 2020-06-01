# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:35:17 2020

Amostra para agupamento de clusters
Base de dados credit_card_data.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO BASEADO EM DENSIDADE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv('CSVs/credit_card_data.csv', header=1)
df['BILL_TOTAL'] = sum(df.iloc[:,12:18].transpose().values)

X = df.iloc[:, [1,25]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=0.377,
    min_samples=5,
    p=2.8,
    metric='minkowski')

predicts = dbscan.fit_predict(X)
unicos, quantidade = np.unique(predicts, return_counts=True)

fig, ax = plt.subplots()
for i in range(len(unicos[unicos>=0])):
    ax.scatter(X[predicts == i,0],
               X[predicts == i,1],
               label = f'Cluster {i+1}')
ax.set_title('Agrupamento por densidade (DBSCAN')
ax.set_ylabel('Dívidas')
ax.set_xlabel('Limite de crédito')
ax.grid(True)
ax.legend()
plt.show()

lst_clients = np.column_stack((df, predicts))
lst_clients = lst_clients[lst_clients[:,26].argsort()]