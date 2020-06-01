# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:31:30 2020

Amostra para agupamento de clusters
Base de dados amostra aleatória make_blobs

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples= 200, centers = 5)

plt.scatter(X[:,0], X[:,1])

kmeans = KMeans(n_clusters = 5)
kmeans.fit(X)

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_

predicts = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=predicts, marker='o')
plt.scatter(centroides[:,0], centroides[:,1], marker = 'x', c='r')

