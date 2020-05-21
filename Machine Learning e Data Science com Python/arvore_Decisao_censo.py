# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:28:14 2020

Amostra para previsão de rendimentos
Base de dados censo_data.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO ÁRVORE DE DECISÃO
"""

# Importa a biblioteca par leitura e manipulação do dataset
import pandas as pd

# Importa a biblioteca para transformar as variáveis categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/censo_data.csv')

# Divide o dataset em previsores e classificadores
previsores = df.iloc[:,0:14].values
classe = df.iloc[:,14].values

# Instancia a classe OneHotEncoder
onehotencoder = ColumnTransformer(
    transformers=[(
        "OneHot",
        OneHotEncoder(),
        [1,3,5,6,7,8,9,13])],
    remainder='passthrough'
    )

# Transforma as variáveis categóricas em Dummies
previsores = onehotencoder.fit_transform(previsores).toarray()

# Instancia a classe LabelEncoder
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

# Exercício: Escalonar somente as features numéricas
scaler = StandardScaler()
# previsores[:,10:13] = scaler.fit_transform(previsores[:,10:13])

"""
O melhor resultado foi sem o OneHotEncoder
Utilizando somente o LabelEncoder e StandardScaler
Accuracy_Score = 0.8057
"""
previsores = scaler.fit_transform(previsores)

# Importa a biblioteca para divisão do dataset em treinamento e teste
from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(
        previsores,
        classe,
        test_size=0.15,
        random_state=0
    )

# Importa a biblioteca Decision_Tree
from sklearn.tree import DecisionTreeClassifier, export

# Instancia a classe GaussianNB
classificador = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0)

# Treina o algoritmo
classificador.fit(previsores_train, classe_train)


export.export_graphviz(classificador,
                       out_file = 'arvore_censo.dot',
                       feature_names = list(df.columns[0:14]),
                       class_names = ['<=50K','>50K'],
                       filled = True,
                       leaves_parallel = True)

# Tira predições do modelo treinado
previsoes = classificador.predict(previsores_test)

# Importa a biblioteca de métricas para comparações
from sklearn.metrics import confusion_matrix, accuracy_score

# Calcula a precisão do modelo
precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test, previsoes)