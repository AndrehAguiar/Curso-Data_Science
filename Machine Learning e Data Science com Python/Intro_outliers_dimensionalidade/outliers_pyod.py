# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:13:10 2020

Detecção de outliers com Boxplot
Base de dados credit_data.csv

@author: TOP Artes
"""

import pandas as pd
from pyod.models.knn import KNN

df = pd.read_csv('CSVs/credit_data.csv')
df = df.dropna()

detector = KNN()
detector.fit(df.iloc[:,1:4])

predicts = detector.labels_
confianca_predicts = detector.decision_scores_

lst_outliers = df.iloc[[i for i, val in enumerate(predicts) if val > 0], :]