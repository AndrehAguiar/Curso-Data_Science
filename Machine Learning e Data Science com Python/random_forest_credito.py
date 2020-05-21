# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:46:36 2020

Amostra para análise de análise de crédito
Base de dados credit_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO ÁRVORE ALEATÓRIA
"""


# Importa a biblioteca para manipulação do dataset
import pandas as pd
import numpy as np

# Importa biblioteca para preencher valores faltantes
from sklearn.impute import SimpleImputer

# Importa biblioteca para padronização
from sklearn.preprocessing import StandardScaler

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/credit_data.csv')

# Define as idades anômolas pela média das normais
df.loc[df.age < 0, 'age'] = df.age[df.age > 0].mean()

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

# Instancia a classe de escalonamento
scaler = StandardScaler()

# Padroniza os valores na mesma escala
previsores = scaler.fit_transform(previsores)

"""
DIVISÃO TREINO E TESTE
"""

# Importa a biblioteca para divisão do dataset em treinamento e teste
from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.25,
        random_state=0
    )

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(
    n_estimators=40,
    criterion='entropy',
    random_state=0)
classificador.fit(previsores_train, classe_train)
previsao = classificador.predict(previsores_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, previsao)
matriz = confusion_matrix(classe_test, previsao)