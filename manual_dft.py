#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 01:26:18 2019

@author: glavigna
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.fftpack as sc

import time

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


def simple_dft(yy,fs,N):
    """
    brief:  Genera la DFT, solamente la primera mitad y en valor absoluto.
    Entradas
    yy: Señal de entrada a convertir en DFT
    ts: Tiempo de sampleo en segundos
    N:  Numero de muestras de la señal
    
    Salidas
    ff: Campo de las frecuencias para poder hacer un grafico
    XX: Espectro de la señal en valor absoluto y solo una mitad.    
    """
    #Calcula la DFT como explicada en las filminas.
    X=np.zeros((N,),dtype=np.complex128);
    for m in range(0,N):
        for n in range(0,N):
            X[m] += signal[n]*np.exp((-2j*(np.pi)*m*n)/N)
    
    ##Espectro para mostrar a la salida es solamente el valor absoluto de la mitad de las muestras.
    XX=(2/N)*np.abs(X[0:N//2])  
    ff=np.linspace(0,fs/2,N//2)
    return ff,XX


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
    """
    XX = (2/N)*np.abs(sc.fft(yy));
    XX = XX[0:N//2];
    ff = np.linspace(0,fs/2,N//2);
     
    return ff,XX

#Cierro todos los graficos por default.
plt.close('all')

fs = 1000
f0 = 2
N  = 512
a0 = 1
p0 = 2*np.pi


(tt,signal) = generador_senoidal(fs , f0 , N , a0 , p0 );


plt.figure(1)
line_hdls = plt.plot(tt,signal)
plt.title('Señal: ' + 'Senoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()

time_before_dft = time.process_time_ns()
(ff,half_dft) = simple_dft(signal,fs,N)
time_after_dft = time.process_time_ns()

print('Tiempo realizando la DFT:{}[s]'.format((time_after_dft-time_before_dft)/1e9))

plt.figure(2)
plt.stem(ff,half_dft)
plt.title('Espectro de la señal haciendo DFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')

time_before_fft = time.process_time_ns()
(ff,half_fft) = simple_fft(signal,fs,N)
time_after_fft = time.process_time_ns()

print('Tiempo realizando la FFT:{}[s]'.format((time_after_fft-time_before_fft)/1e9))

plt.figure(3)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')

