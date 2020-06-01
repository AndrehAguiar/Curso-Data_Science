# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:19:20 2020

Comparação dos algorítimos de agrupamento

K-Means X Hieráquico X DBSCAN

@author: TOP Artes
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets

X, y = datasets.make_moons(
    n_samples = 1500,
    noise=0.09)
plt.scatter(X[:,0], X[:,1], s=5)

kmeans = KMeans(n_clusters = 2)
predicts_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1], s=5, c=predicts_kmeans)

hc = AgglomerativeClustering(
    n_clusters = 2,
    affinity = 'euclidean',
    linkage = 'ward')
predicts_hc = hc.fit_predict(X)
plt.scatter(X[:,0], X[:,1], s=5, c=predicts_hc)

dbscan = DBSCAN(eps=0.1)
predicts_dbscan = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1], s=5, c=predicts_dbscan)