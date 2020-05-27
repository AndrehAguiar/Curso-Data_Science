# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:04:12 2020

Estruturação de Rede Neural com pesos aleatórios

@author: TOP Artes
MODELO CLASSIFICAÇÃO PyBrain
"""

from pybrain.structure import (
    FeedForwardNetwork,
    FullConnection,
    LinearLayer,
    SigmoidLayer,
    BiasUnit
    )

rede = FeedForwardNetwork()

# Define a estrutura da rede
camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

# Define os módulos da rede
rede.addInputModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addOutputModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

# Define as conecções entre as camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

# Estrutura de camadas da rede
print(f'Estrutura de camadas da rede {rede}')
# Pesos aleatórios da camada de entrada
print(f'Pesos aleatórios da camada de entrada {entradaOculta.params}')
# Valores de saída
print(f'Valores de saída {ocultaSaida.params}')
# Pesos de entrada da camada Bias até a camada oculta
print(f'Pesos de entrada da camada Bias até a camada oculta {biasOculta.params}')
# Pesos de saída da camada Bias até a saída
print(f'Pesos de saída da camada Bias até a saída {biasSaida.params}')