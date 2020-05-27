# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:05:32 2020

@author: TOP Artes
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('CSVs/credit_data.csv')
df.loc[df.age < 0, 'age']= df.age[df.age > 0].mean()
previsores = df.iloc[:, 1:4].values
classe = df.iloc[:,4].values
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy='mean')
imputer = imputer.fit(previsores)
previsores = imputer.transform(previsores)
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('rf_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

resultado_svm = svm.score(previsores, classe)
resultado_rf = random_forest.score(previsores, classe)
resultado_mlp = mlp.score(previsores, classe)

novo_registro = [[50000,40,5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1,1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1,3)

resposta_svm = svm.predict(novo_registro)
resposta_rf = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)