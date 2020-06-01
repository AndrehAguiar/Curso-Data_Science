# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:34:45 2020

Amostra para agupamento de clusters
Base de dados credit_card_data.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO HIERÁRQUICO
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

X = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]
y = [1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]

plt.scatter(X, y)

arr_base = np.array([[X[i], y[i]] for i in range(len(X))])

scaler = StandardScaler()
arr_base = scaler.fit_transform(arr_base)

fig, ax = plt.subplots()
dendrograma = dendrogram(linkage(arr_base, method='ward'))
ax.set_title('Dendrograma')
ax.set_xlabel('Pessoas')
ax.set_ylabel('Distância Euclidiana')
plt.show()

hc = AgglomerativeClustering(
    n_clusters = 3,
    affinity = 'euclidean',
    linkage = 'ward')
predicts = hc.fit_predict(arr_base)

fig, ax = plt.subplots()
for i in range(len(np.unique(predicts))):
    plt.scatter(arr_base[predicts == i, 0],
                arr_base[predicts == i, 1],
                label = f'Cluster {i+1}')
    
ax.set_title('Agrupamento')
ax.set_xlabel('Idade')
ax.set_ylabel('Salário')
ax.legend()
plt.show()
    
