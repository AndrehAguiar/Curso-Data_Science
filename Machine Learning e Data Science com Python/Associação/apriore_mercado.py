# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:52:26 2020

Amostra didática para associação de produtos
Base de dados mercado.csv

@author: TOP Artes

PRE-PROCESSAMENTO
REGRAS DE ASSOCIAÇÃO
"""

import pandas as pd
from apyori import apriori

df = pd.read_csv('CSVs/mercado.csv', header=None)

transacoes = [[str(df.values[j, i]) for i in range(len(df.columns))] for j in range(len(df))]
regras = apriori(transacoes,
                 min_support = 0.3,
                 min_confidence = 0.8,
                 min_lift = 2,
                 max_lenght = 2)

lst_resultados = [list(x) for x in list(regras)]
lst_itemsets = [[list(x) for x in lst_resultados[j][2]] for j in range(len(lst_resultados))]
