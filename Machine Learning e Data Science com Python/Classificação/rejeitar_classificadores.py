# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:05:32 2020

@author: TOP Artes
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('rf_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))


novo_registro = [[50000,40,5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1,1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1,3)

resposta_svm = svm.predict(novo_registro)
resposta_rf = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)

probabilidade_svm = svm.predict_proba(novo_registro)
confianca_svm = probabilidade_svm.max()

probabilidade_rf = random_forest.predict_proba(novo_registro)
confianca_rf = probabilidade_rf.max()

probabilidade_mlp = mlp.predict_proba(novo_registro)
confianca_mlp = probabilidade_mlp.max()

paga = 0
nao_paga = 0
confianca_minima = 0.98

"""
Tipos de decisão:
    Unanimidade: 100% dos votos
    Maioria simples: 50% votos + 1
    Pluralidade: maioria dos votos (34%, 33%, 33%)
"""
if confianca_svm >= confianca_minima:
    if resposta_svm[0] == 1:
        paga+=1
    else:
        nao_paga+=1
        
if confianca_rf >= confianca_minima: 
    if resposta_rf[0] == 1:
        paga+=1
    else:
        nao_paga+=1
    
if confianca_mlp >= confianca_minima:
    if resposta_mlp[0] == 1:
        paga+=1
    else:
        nao_paga+=1
    
if paga > nao_paga:
    print('Cliente pagará o empréstimo')
elif paga == nao_paga:
    print('Resultado empatado')
else:
    print('Cliente não pagará o empréstimo')