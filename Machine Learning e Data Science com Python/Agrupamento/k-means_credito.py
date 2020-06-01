# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:31:30 2020

Amostra para agupamento de clusters
Base de dados credit_card_data.csv

@author: TOP Artes

ANÁLISE DOS DADOS PARA AGRUPAMENTO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('CSVs/credit_card_data.csv', header=1)

# Cria o somatório de todas as dívidas
df['TOTAL_BILL'] = sum(df.iloc[:,12:18].transpose().values)

# Seleciona o limite de crédito e o total de dívidas
X = df.iloc[:,[1,25]]

# Escalona os valores para agrupamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define o somatório das distâncias entre os clusters
# Within-cluster sum of squares
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.set_title('The elbow Method')
ax.set_ylabel('Avg Within-Cluster distance to centroid (WCSS)')
ax.set_xlabel('Number of clusters K')
ax.plot(range(1,11),wcss)
ax.grid(True)
plt.show()

kmeans = KMeans(n_clusters = 4, random_state = 0)
predicts = kmeans.fit_predict(X)
X = scaler.inverse_transform(X)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=predicts)
ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='r')
ax.set_title('Agrupamento de clientes por limite de crédito')
ax.set_ylabel('Total da dívida acumulada')
ax.set_xlabel('Limite de crédito')
ax.grid(True)
# ax.legend()
plt.show()

lst_clients = np.column_stack((df, predicts))
lst_clients = lst_clients[lst_clients[:,26].argsort()]