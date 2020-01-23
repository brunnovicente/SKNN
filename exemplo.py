# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:51:21 2019

@author: bruno
"""
import BaseDados as BD
from SemiKNN import Semi_Supervised_KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

""" Coleta e Normalização dos dados """
sca = MinMaxScaler()
dados = BD.base_qualquer('d:/basedados/mnist64.csv')
Y = dados[1] # Para ficar com classe 0, 1 , 2
#X = sca.fit_transform(dados[0])
X = dados[0]
L, U, y, yu = train_test_split(X, Y, train_size=0.3, test_size=0.7, stratify=Y)

""" Instanciação do objeto do KNN Semissupervisionado """
semiKNN = Semi_Supervised_KNN()

""" Rotulação as amostras de U usando K=5 """
preditas = semiKNN.classificar(L, U, y, k=5)

""" Calculando a acurácia """
acc = accuracy_score(yu, preditas)

print('Acurácia: ', acc)

