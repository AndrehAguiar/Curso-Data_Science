# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:53:45 2020

Amostra para análise de risco de crédito
Base de dados risco_credito.csv

@author: TOP Artes

PRE-PROCESSAMENTO
PREDIÇÃO REGRESSÃO LOGÍSTICA
"""
# Importa a biblioteca par leitura e manipulação do dataset
import pandas as pd

# Faz a leitura do DataFrame
df = pd.read_csv('CSVs/risco_credito.csv')

df = df[df['c#risco'] != 'moderado']

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

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores, classe)

print(classificador.intercept_)
print(classificador.coef_)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])
probabilidade = classificador.predict_proba([[0,0,1,2],[3,0,0,0]])
print(resultado)
print(probabilidade)