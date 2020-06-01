# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:13:10 2020

Detecção de outliers com Boxplot
Base de dados credit_data.csv

@author: TOP Artes
"""

import pandas as pd

df = pd.read_csv('CSVs/credit_data.csv')
df = df.dropna()

import matplotlib.pyplot as plt

# Outliers IDADE
plt.boxplot(df.iloc[:,2], showfliers = True)
outliers = df[(df.age < -20)]

# Outliers LOAN
plt.boxplot(df.iloc[:,3], showfliers = True)
plt.grid(True)
outliers = df[(df.loan > 13000)]
