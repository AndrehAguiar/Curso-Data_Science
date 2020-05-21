# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:55:17 2020
Amostra para classificação de crédito
Base de dados credit_data.csv
@author: TOP Artes
"""
# Importa a biblioteca para manipulação do dataset
import pandas as pd

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/credit_data.csv')

# Identifica estatísticas básicas dp dataset
df.describe()

# Localiza dados especificados Idade menor que 0 e nulos
df.loc[df['age'] < 0]
df.loc[df['age'].isna()]

# Apaga a coluna inteira
### df.drop('age', 1, inplace=True)

# Apaga somente as linhas especificadas
### df.drop(df[df.age < 0].index, inplace=True)

# Preecher os valores manualmente
# preencher os valores com a média
df.mean()
df.age.mean()
df.age[df.age > 0].mean()

# Define as idades anômolas pela média das normais
df.loc[df.age < 0, 'age'] = df.age[df.age > 0].mean()

# Identifica valores nulos
pd.isnull(df.age)
df.loc[pd.isnull(df.age)]

# Define as idades não informadas pela média das normais
### df.loc[df.age.isna(), 'age'] = df.age[df.age > 0].mean()

# Define as features para previsores
previsores = df.iloc[:,1:4].values
classe = df.iloc[:, 4].values

import numpy as np

# Library para preencher valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy='mean'
    )

# Treina o algoritmo com os valores existentes
imputer = imputer.fit(previsores[:, 0:3])

# Insere os valores nas células com valores faltantes
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# Importa biblioteca para padronização
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Padroniza os valores na mesma escala
previsores = scaler.fit_transform(previsores)



"""
Created on Sun May 17 11:03:17 2020
Amostra para previsão de rendimentos
Base de dados censo_data.csv
@author: TOP Artes
"""

# Lê o dataset credit_data e define como df
df = pd.read_csv('CSVs/censo_data.csv')

# Divide o dataset em previsores e classificadores
previsores = df.iloc[:,0:14].values
classe = df.iloc[:,14].values

# Importa a biblioteca para transformar as variáveis categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Instancia a classe LabelEncoder dos previsores
labelencoder_previsores = LabelEncoder()


# Instancia a classe OneHotEncoder
onehotencoder = ColumnTransformer(
    transformers=[(
        "OneHot",
        OneHotEncoder(),
        [1,3,5,6,7,8,9,13])],
    remainder='passthrough'
    )

previsores = onehotencoder.fit_transform(previsores).toarray()

# Instancia a classe LabelEncoder das classes
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)