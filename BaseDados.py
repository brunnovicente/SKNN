# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.datasets import reuters
from keras.datasets import mnist
from sklearn.datasets import load_digits

def vetorizar_sequencias(sequencias, dimensao = 10000):
    resultados = np.zeros((len(sequencias), dimensao))
    for i, sequencia in enumerate(sequencias):
        resultados[i, sequencia] = 1.
    return resultados

def base_qualquer(caminho):
    dados = pd.read_csv(caminho)
    X = dados.drop(['classe'], axis=1).values
    Y = dados['classe'].values
    return (X,Y)

def base_iris(caminho = ''):
    dados = pd.read_csv(caminho+'iris.csv')
    X = dados.drop(['classe'], axis=1).values
    Y = dados['classe'].values
    return (X,Y)

def base_mnist64():
    digitos = load_digits()
    X = digitos.data
    Y = digitos.target
    return (X, Y)

def base_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    entradas = x_train.tolist()
    saidas = y_train.tolist()
    X = np.array(entradas).reshape(60000,784)
    Y = np.array(saidas, dtype=np.int64)
    return (X, Y)

def base_letras(caminho = ''):
    dados = pd.read_csv(caminho + 'letras.csv')
    #X,Xt, Y, Yt = train_test_split(dados.drop(['classe'], axis=1).values,dados['classe'].values, train_size=0.75, test_size=0.25, stratify=dados['classe'].values)
    X = dados.drop(['classe'], axis=1).values
    Y = dados['classe'].values
    return (X,Y)

def base_mnist_fashion(caminho = ''):
    dados = pd.read_csv(caminho + 'fashion_mnist.csv')
    X = dados.drop(['classe'], axis=1).values
    Y = dados['classe'].values
    return (X,Y)
    
def base_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    x_train = vetorizar_sequencias(x_train, dimensao=5000)
    x_test = vetorizar_sequencias(x_test, dimensao=5000)
    
    entradas = x_train.tolist()
    entradas.extend(x_test.tolist())
    
    saidas = y_train.tolist()
    saidas.extend(y_test.tolist())
    
    X = np.array(entradas)
    Y = np.array(saidas)
    
    return (X,Y)

def base_reuters():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=500)
    x_train = vetorizar_sequencias(x_train, dimensao=500)
    x_test = vetorizar_sequencias(x_test, dimensao=500)
    
    entradas = x_train.tolist()
    entradas.extend(x_test.tolist())
    
    saidas = y_train.tolist()
    saidas.extend(y_test.tolist())
    
    X = np.array(entradas)
    Y = np.array(saidas)
    
    return (X,Y)