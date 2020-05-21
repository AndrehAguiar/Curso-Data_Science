# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:47:38 2020

Itrodução de regras com Orange
Base de dados credit_data.csv

@author: TOP Artes
"""
import Orange
df = Orange.data.Table('CSVs/credit_data.csv')
df.domain

df_dividida = Orange.evaluation.testing.sample(df, n=0.25)

df_train = df_dividida[1]
df_test = df_dividida[0]
len(df_train)
len(df_test)

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(df_train)

for regras in classificador.rule_list:
    print(regras)
    
resultado = Orange.evaluation.testing.TestOnTestData(df_train, df_test,[lambda testdata: classificador])

print(Orange.evaluation.CA(resultado))