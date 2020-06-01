# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:35:17 2020

Amostra para agupamento de clusters
Base de dados didáica

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO BASEADO EM DENSIDADE
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


X=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]

plt.scatter(X, y)
arr_base = np.array([[X[i], y[i]] for i in range(len(X))])

scaler = StandardScaler()
arr_base = scaler.fit_transform(arr_base)

dbscan = DBSCAN(eps = 0.95, min_samples = 2)
dbscan.fit(arr_base)
predicts = dbscan.labels_

fig, ax = plt.subplots()
ax.scatter(arr_base[:,0],
           arr_base[:,1],
           c=predicts)
ax.set_title('Agrupamento por densidade (DBSCAN')
ax.set_xlabel('Dívidas')
ax.set_ylabel('Limite de crédito')
ax.grid(True)
ax.legend()
plt.show()