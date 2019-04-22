#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 02:23:33 2019

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
    
    delta_f = (fs/2)/(N//2) ;
    
    XX = (2/N)*np.abs(sc.fft(yy));
    XX = XX[0:N//2];
    ff = np.linspace(0,(fs/2)-delta_f,N//2);
     
    return ff,XX

def center_of_mass_fft(ff,half_fft):
    return ((np.sum(np.multiply(ff,half_fft)))/np.sum(half_fft))

#Cierro todos los graficos por default.
plt.close('all')

fs = 1000
ts = 1/fs
f0 = fs/4
delta = 0.5
N  = 1000
a0 = 1
p0 = 0

#Generar la senoidal
(tt,signal) = generador_senoidal(fs , f0 + delta , N , a0 , p0 );

signal = np.pad(signal,(0,9*N),'constant')
tt = np.linspace(0,(len(signal)-1)*ts, len(signal)).flatten()

plt.figure(1)
line_hdls = plt.plot(tt,signal)
plt.title('Señal: ' + 'Senoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()


(ff,half_fft) = simple_fft(signal,fs,len(signal))

plt.figure(2)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo DFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()

max_mag_value = np.amax(half_fft);
max_freq_value = np.where(half_fft == max_mag_value)

print('Frecuencia Target es: {} Hz'.format(f0 + delta))
print('Frecuencia donde se encuentra el maximo: {} Hz'.format(round(ff[max_freq_value[0][0]],4)))
print('Centro de masa de la frecuencia {}'.format(round(center_of_mass_fft(ff,half_fft),4)))
print('Paso de frecuencia:{} Hz'.format((ff[1]-ff[0])))
#print('{}'.format(sum_test))




