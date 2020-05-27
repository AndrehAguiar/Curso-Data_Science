# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:28:18 2020

@author: TOP Artes
"""
import pandas as pd
import numpy as np

# Importa biblioteca para preencher valores faltantes
from sklearn.impute import SimpleImputer

# Importa biblioteca para padronização
from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/credit_data.csv')

# Define as idades anômolas pela média das normais
df.loc[df.age < 0, 'age'] = df.age[df.age > 0].mean()

# Identifica valores nulos
pd.isnull(df.age)
df.loc[pd.isnull(df.age)]

# Define as features para previsores
previsores = df.iloc[:,1:4].values
classe = df.iloc[:, 4].values

# Instancia a classe Imputer
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy='mean'
    )

# Treina o algoritmo com os valores existentes
imputer = imputer.fit(previsores)

# Insere os valores nas células com valores faltantes
previsores = imputer.transform(previsores)

# Istancia a classe StandardScaler para escalonamento dos valores
scaler = StandardScaler()
# Padroniza os valores na mesma escala
previsores = scaler.fit_transform(previsores)

classificadorSVM = SVC(kernel='rbf', C=2.0, probability=True)
classificadorSVM.fit(previsores, classe)

classificadorRF = RandomForestClassifier(n_estimators=40, criterion='entropy')
classificadorRF.fit(previsores, classe)

classificadorMLP = MLPClassifier(
    verbose=True, max_iter=1000, tol = 0.000010,
    solver = 'adam', hidden_layer_sizes=(100), activation='relu',
    batch_size=200, learning_rate_init=0.001)

classificadorMLP.fit(previsores, classe)

import pickle
pickle.dump(classificadorSVM, open('svm_finalizado.sav', 'wb'))
pickle.dump(classificadorRF, open('rf_finalizado.sav', 'wb'))
pickle.dump(classificadorMLP, open('mlp_finalizado.sav', 'wb'))














