# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:43:01 2020

Amostra para análise de risco de crédito
Base de dados risco_credito.csv

@author: TOP Artes

PRE-PROCESSAMENTO
CLASSIFICAÇÃO ÁRVORE DE DECISÃO
"""
# Importa a biblioteca par leitura e manipulação do dataset
import pandas as pd

# Faz a leitura do DataFrame
df = pd.read_csv('CSVs/risco_credito.csv')

# Divide a base em previsores e classificadores
previsores = df.iloc[:,0:4].values
classe = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

# Instancia a classe LabelEncoder dos previsores
labelencoder = LabelEncoder()

# Codifica as variáveis categórias
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

# Importa a biblioteca DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export

# Instancia a classe DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy')

# Treina o algoritmo
classificador.fit(previsores, classe)
print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia','divida','garantias', 'renda'],
                       class_names = ['alto','moderado','baixo'],
                       filled = True,
                       leaves_parallel = True)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)


