#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:26:21 2019

@author: glavigna
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def generador_ruido(fs,N,media = 0, varianza = 1):
    
    # tiempo de muestreo
    ts = 1/fs     
    #Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
    #Flatten convierte a un array de 1 dimensión.
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Concatenación de matrices:
    # guardaremos las señales creadas al ir poblando la siguiente matriz vacía
    signal = np.array([], dtype=np.float).reshape(N,0)
    #Generar señal de media X y desvio estandar sacada de numpy.
    signal = np.random.normal(media,np.sqrt(varianza),N);
        
    return tt,signal

fs          = 1000;
N           = 1000;
media       = 0;
varianza    = 1;

(tt,signal) = generador_ruido(fs,N,media,varianza);

plt.figure(1)
line_hdls = plt.plot(tt,signal)
plt.title('Señal: ' + 'Ruido Mu:{} Sigma:{}'.format(media,np.sqrt(varianza)))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
    