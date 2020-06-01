# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:55:29 2020

Amostra para previsão de rendimentos
Base de dados censo_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
REDUÇÃO DE DIMENSIONALIDADE PCA
"""

# Importa a biblioteca par leitura e manipulação do dataset
import pandas as pd
import collections

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


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divide a base em treino e teste
previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.15,
        random_state=0
    )

"""
Implementar classificador aqui
"""
from sklearn.decomposition import PCA

pca = PCA(n_components = 6)
previsores_train = pca.fit_transform(previsores_train)
previsores_test = pca.transform(previsores_test)
comonents = pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestClassifier

classificador = RandomForestClassifier(
    n_estimators = 40,
    criterion = 'entropy',
    random_state = 0)

classificador.fit(previsores_train, classe_train)
previsao = classificador.predict(previsores_test)
precisao = accuracy_score(classe_test, previsao)