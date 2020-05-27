# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:17:13 2020

Amostra para classificação de crédito
Base de dados credit_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO KERAS
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
imputer = imputer.fit(previsores[:, 0:3])

# Insere os valores nas células com valores faltantes
previsores = imputer.transform(previsores)

# Istancia a classe StandardScaler para escalonamento dos valores
scaler = StandardScaler()
# Padroniza os valores na mesma escala
previsores = scaler.fit_transform(previsores)

"""
DIVISÃO TREINO E TESTE
"""

previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.25,
        random_state=0
    )

"""
Implementar classificador aqui
"""
import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
classificador.add(
    Dense(
        units=2,
        activation='relu',
        input_dim = 3))
classificador.add(
    Dense(
        units=2,
        activation='relu'))
classificador.add(
    Dense(
        units=1,
        activation='sigmoid'))
classificador.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

classificador.fit(
    previsores_train,
    classe_train,
    batch_size = 10,
    epochs = 100)

previsao = classificador.predict(previsores_test)
previsao = (previsao > 0.5)

# Confirma métricas para comparação
precisao = accuracy_score(classe_test, previsao)
matriz = confusion_matrix(classe_test, previsao)