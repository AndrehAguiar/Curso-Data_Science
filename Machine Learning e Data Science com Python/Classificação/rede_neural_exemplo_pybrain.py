# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:44:04 2020

@author: TOP Artes
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer, SoftmaxLayer

rede = buildNetwork(2, 3, 1,
                    outclass = SoftmaxLayer,
                    hiddenclass = SigmoidLayer,
                    bias = False)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])



# Configuração padrão DATA (XOR)
rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2,1)
base.addSample((0,0), (0, ))
base.addSample((0,1), (1, ))
base.addSample((1,0), (1, ))
base.addSample((1,1), (0, ))

print(base['input'])
print(base['target'])

treinamento = BackpropTrainer(rede,
                              dataset = base,
                              learningrate=0.01,
                              momentum=0.06)

for i in range(1, 30000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s " % erro)
        
print(rede.activate([0,0]))
print(rede.activate([1,0]))
print(rede.activate([0,1]))
print(rede.activate([1,1]))