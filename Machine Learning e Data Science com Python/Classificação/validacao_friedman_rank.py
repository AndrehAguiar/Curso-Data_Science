# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:06:49 2020

@author: TOP Artes
"""


import pandas as pd
df = pd.read_csv('CSVs/dados_testes.csv', sep=';', decimal=',')

import Orange

import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
avranks =  df.mean()
names = [avranks.index[i]+' - '+str(round(avranks[i], 3)) for i in range(len(avranks))]
cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets

title = f'Friedman-Nemenyi (CD = {round(cd, 3)})'

Orange.evaluation.graph_ranks(
    avranks, names,
    cd=cd,
    width=6,
    textspace=1.5)

plt.title(title)
plt.show()