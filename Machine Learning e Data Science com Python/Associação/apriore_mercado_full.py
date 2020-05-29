# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:52:26 2020

Amostra didática para associação de produtos
Base de dados mercado1.csv

@author: TOP Artes

PRE-PROCESSAMENTO
REGRAS DE ASSOCIAÇÃO
"""

import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

df = pd.read_csv('CSVs/mercado1.csv', header=None)

transacoes = [[str(df.values[j, i]) for i in range(len(df.columns))] for j in range(len(df))]
# support = produtos vendidos
# 4 vezes ao dia
# 7 dias da semana (base de vendas semanal)
# dividivo pelo tamanho da base
regras = apriori(transacoes,
                 min_support = 0.003,
                 min_confidence = 0.2,
                 min_lift = 3,
                 max_lenght = 2)

lst_resultados = [list(x) for x in list(regras)]
lst_itemsets = [[list(x) for x in lst_resultados[j][2]] for j in range(len(lst_resultados))]
