# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:12:44 2020
Amostra para previsão de rendimentos
Base de dados censo_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO Rede Neural(MLP)
"""

# Importa a biblioteca par leitura e manipulação do dataset
import pandas as pd

# Importa a biblioteca para transformar as variáveis categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Importa a biblioteca para divisão do dataset em treinamento e teste
from sklearn.model_selection import train_test_split

# Importa a biblioteca de métricas para conclusão
from sklearn.metrics import confusion_matrix, accuracy_score

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/censo_data.csv')

# Divide o dataset em previsores e classificadores
previsores = df.iloc[:,0:14].values
classe = df.iloc[:,14].values

# Instancia a classe LabelEncoder das classes
labelencoder = LabelEncoder()
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 13])

classe = labelencoder.fit_transform(classe)


# Instancia a classe OneHotEncoder
onehotencoder = ColumnTransformer(
    transformers=[(
        "OneHot",
        OneHotEncoder(),
        [1,3,5,6,7,8,9,13])],
    remainder='passthrough'
    )

previsores = onehotencoder.fit_transform(previsores).toarray()

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divide a base em treino e teste
previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.25,
        random_state=0
    )

"""
Implementar classificador aqui
"""
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(
    verbose = True,
    max_iter = 1000,
    tol=10e-6,
    hidden_layer_sizes=(100))
classificador.fit(previsores_train, classe_train)
previsao = classificador.predict(previsores_test)

precisao = accuracy_score(classe_test, previsao)
matriz = confusion_matrix(classe_test, previsao)
