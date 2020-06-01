# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:34:45 2020

Amostra para agupamento de clusters
Base de dados credit_card_data.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO HIERÁRQUICO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CSVs/credit_card_data.csv', header = 1)

df['BILL_TOTAL'] = sum(df.iloc[:,12:18].transpose().values)

X = df.iloc[:, [1,25]]

scaler = StandardScaler()
X = scaler.fit_transform(X)

fig, ax = plt.subplots()
dendrograma = dendrogram(linkage(X, method='ward'))
ax.set_title('Dendrograma')
ax.set_xlabel('Pessoas')
ax.set_ylabel('Distância Euclidiana')
plt.show()

hc = AgglomerativeClustering(
    n_clusters = 3,
    affinity = 'euclidean',
    linkage = 'ward')

predicts = hc.fit_predict(X)
fig, ax = plt.subplots()
for i in range(len(np.unique(predicts))):
    plt.scatter(X[predicts == i, 0],
                X[predicts == i, 1],
                label = f'Cluster {i+1}')
    
ax.set_title('Agrupamento')
ax.set_xlabel('Idade')
ax.set_ylabel('Salário')
ax.legend()
plt.show()
    
