# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:10:40 2020

Itrodução de regras com Orange
Base de dados risco_credito.csv

@author: TOP Artes
"""
import Orange

df = Orange.data.Table('CSVs/risco_credito.csv')
df.domain

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(df)

for regras in classificador.rule_list:
    print(regras)


# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15    
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'],
                           ['ruim', 'alta','adequada','0_15']])

for i in resultado:
    print(df.domain.class_var.values[i])