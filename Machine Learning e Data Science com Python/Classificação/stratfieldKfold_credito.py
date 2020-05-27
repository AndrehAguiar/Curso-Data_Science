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

# Importa a biblioteca de calassificação
from sklearn.naive_bayes import GaussianNB

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

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import Orange
df = Orange.data.Table('CSVs/credit_data.csv')

from sklearn.metrics import accuracy_score, confusion_matrix
resultados_mean = []
for i in range(0,30):
    kfold = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=i)
    
    resultados = []
    matrizes = []
    
    # Modelos SkLearn
    # for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(previsores.shape[0], 1))):
        # classificador = GaussianNB()
        # classificador = DecisionTreeClassifier()
        # classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        # classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        # classificador = LogisticRegression()        
        # classificador = SVC(kernel='poly', random_state=1, C=2.0)
        # classificador = MLPClassifier(verbose = True, max_iter=1000, tol = 1e-6, solver = 'adam', hidden_layer_sizes=(100), activation='relu')
        
        

        # df_dividida = Orange.evaluation.testing.sample(df, n=0.25)        
        # df_train = df_dividida[1]
        # df_test = df_dividida[0]

        # classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        # previsao = classificador.predict(previsores[indice_teste])
        # precisao = accuracy_score(classe[indice_teste], previsao)
        # matrizes.append(confusion_matrix(classe[indice_teste], previsao))
    
    # Modelo Orange CN2
    for indice_treinamento, indice_teste in kfold.split(df,
                                                    np.zeros(shape=(2000,1))):
        cn2_learner = Orange.classification.rules.CN2Learner()
        classificador = cn2_learner(df[indice_treinamento])
            
        previsoes = classificador(df[indice_teste])
        
        precisao = accuracy_score(df.Y[indice_teste], previsoes)
    
    resultados.append(precisao)
    result = np.asarray(resultados)
    resultados_mean.append(result.mean())

resultados_mean = np.asarray(resultados_mean)
for i in range(30):
    print(str(resultados_mean[i]).replace('.',','))

matriz_final = np.mean(matrizes, axis = 0)
resultados = np.asarray(resultados)
resultados.mean()
resultados.std()