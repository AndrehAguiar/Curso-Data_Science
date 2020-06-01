# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:13:10 2020

Detecção de outliers com Boxplot
Base de dados credit_data.csv

@author: TOP Artes
"""

import pandas as pd

import matplotlib.pyplot as plt

df_credito = pd.read_csv('CSVs/credit_data.csv')
df_credito = df.dropna()

# Outliers LIMITE / IDADE
plt.scatter(
    df_credito.iloc[:,1],
    df_credito.iloc[:,2])
outliers = df_credito[(df_credito.age < -20)]

# Outliers LIMITE / LOAN
plt.scatter(
    df_credito.iloc[:,1],
    df_credito.iloc[:,3])
plt.grid(True)

# Outliers IDADE / LOAN
plt.scatter(
    df_credito.iloc[:,2],
    df_credito.iloc[:,3])

# Tratando outliers
df_credito.loc[df_credito.age < 0, 'age'] = df_credito['age'][df_credito.age > 0].mean()

df_censo = pd.read_csv('CSVs/censo_data.csv')

# Outliers Idade / Final Weight
plt.scatter(
    df_censo.iloc[:,0],
    df_censo.iloc[:,2])
