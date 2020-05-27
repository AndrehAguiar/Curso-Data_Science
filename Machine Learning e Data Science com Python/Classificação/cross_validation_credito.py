# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:16:55 2020

Amostra para classificação de crédito
Base de dados credit_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CROSS VALIDATION
"""

# Importa a biblioteca para manipulação do dataset
import pandas as pd
import numpy as np

# Importa biblioteca para preencher valores faltantes
from sklearn.impute import SimpleImputer

# Importa biblioteca para padronização
from sklearn.preprocessing import StandardScaler

# Importa a biblioteca para divisão do dataset em treinamento e teste
from sklearn.model_selection import train_test_split

# Importa a biblioteca de métricas para conclusão
from sklearn.metrics import confusion_matrix, accuracy_score

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

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
resultados = cross_val_score(
    classificador, previsores, classe, cv = 10)
resultados.mean()
resultados.std()