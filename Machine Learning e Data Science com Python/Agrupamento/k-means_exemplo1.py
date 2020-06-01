# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:31:30 2020

Amostra para agupamento de clusters
Base de dados modelo didático

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]


arr_base = np.array([[X[i], y[i]] for i in range(len(X))])

scaler = StandardScaler()
arr_base = scaler.fit_transform(arr_base)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(arr_base)

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_

plt.scatter(arr_base[:,0], arr_base[:,1], c=rotulos)
plt.scatter(centroides[:,0], centroides[:,1], marker = 'x', c='r')

