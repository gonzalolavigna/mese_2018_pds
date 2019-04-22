#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:30:07 2019

@author: glavigna
"""

from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#Estos es para hacer lo del ipython
#%matplotlib qt5

N   = 1024
Fs  = 400000
fsig = 500
avg = 10
an  = 0.1

tt = np.linspace(0,N/Fs,N,endpoint = 'False')


#Genero una senoidal de frecuencia fsig
xsin = np.sin(2*np.pi*fsig*tt)
#Genero señal de ruido con amplitud unitaria
xn = an*np.random.randn(len(tt));
#Sumo al ruido la señal de ruido con la señalo
x = xsin + xn


plt.plot(tt,x)

#Graficar el espectro resultante.
#Tarea para el hogar.


#Lo construimos a traves der la respuesta al impulso. Al promediador
h = (1/avg)*np.ones(avg)

#  y = x conv h

y = signal.convolve(x,h)
y = y[0:-(avg-1)]


#Tareas para el hogar!! de la clase 05 de PDF
#1 Graficar el espectro de la señal de entrada y de la señal de salida.
#2 Espetro de la respuesta al importa al impulso en frecuencia.
#3 Estudiar promediado
#4 Comparar orden par y orden impar
#5 Estudiar el fecto de zeros intercalas h= [1,1,1] => [1 0 1]
