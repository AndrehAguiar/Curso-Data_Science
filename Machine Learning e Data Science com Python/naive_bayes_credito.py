# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:59:16 2020

Amostra para análise de análise de crédito
Base de dados credit_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO NAIVE BAYES
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
imputer = imputer.fit(previsores[:, 0:3])

# Insere os valores nas células com valores faltantes
previsores[:,0:3] = imputer.transform(previsores[:,0:3])
scaler = StandardScaler()

# Padroniza os valores na mesma escala
previsores = scaler.fit_transform(previsores)

# Importa a biblioteca para divisão do dataset em treinamento e teste
from sklearn.model_selection import train_test_split

# Divide a base em treino e teste
previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.25,
        random_state=0
    )


# Importa a biblioteca naive_bayes
from sklearn.naive_bayes import GaussianNB

# Instancia a classe GaussianNB
classificador = GaussianNB()

# Treina o algoritmo
classificador.fit(previsores_train, classe_train)

# Tira predições do modelo treinado
previsoes = classificador.predict(previsores_test)

# Importa a biblioteca de métricas para comparações
from sklearn.metrics import confusion_matrix, accuracy_score

# Calcula a precisão do modelo
precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test, previsoes)

