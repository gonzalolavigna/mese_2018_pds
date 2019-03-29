#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:32:59 2019

@author: glavigna
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.fftpack as sc

def generador_senoidal (fs, f0, N, a0=1, p0=0):
    
    ts = 1/fs # tiempo de muestreo    
    
    #Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
    #Flatten convierte a un array de 1 dimensión.
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Concatenación de matrices:
    # guardaremos las señales creadas al ir poblando la siguiente matriz vacía
    signal = np.array([], dtype=np.float).reshape(N,0)
    
    #Genero la senoidal
    signal = a0 * np.sin(2 * np.pi * f0 * tt + p0);
   
    return tt,signal

def simple_fft(yy,fs,N):
    """
    brief:  Genera la DFT, pero utilizando la FFT como algoritmo
    Entradas
    yy: Señal de entrada a convertir en DFT
    ts: Tiempo de sampleo en segundos
    N:  Numero de muestras de la señal
    
    Salidas
    ff: Campo de las frecuencias para poder hacer un grafico
    XX: Espectro de la señal en valor absoluto y solo una mitad.
        Magnitud Normalizada.
    """
    XX = (2/N)*np.abs(sc.fft(yy));
    XX = XX[0:N//2];
    ff = np.linspace(0,fs/2,N//2);
     
    return ff,XX


fs = 1000
f0 = fs/4
delta = 0
N  = 1000
a0 = 1
p0 = 0

#Generar la senoidal
(tt,signal) = generador_senoidal(fs , f0 + delta , N , a0 , p0 );


plt.figure(1)
line_hdls = plt.plot(tt,signal)
plt.title('Señal: ' + 'Senoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()


(ff,half_fft) = simple_fft(signal,fs,N)

plt.figure(2)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo DFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()

print('Magnitud Frecuencia Central:{}'.format(half_fft[int(f0)]))
print('Magnitud Frecuencia Adyacente:{}'.format(half_fft[int(f0+1)]))

spread = np.sum(np.concatenate((half_fft[:int(f0)],half_fft[int(f0)+1:]),axis=0))
print('Magnitud Resto de las frecuencias:{}'.format(spread))
#print('{}'.format(sum_test))




