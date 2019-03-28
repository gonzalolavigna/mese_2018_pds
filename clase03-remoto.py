#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:04:46 2019

@author: glavigna
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sc
import scipy.signal as signal


#Es un valor asociado a la memoria --> Conviene elegir siempre modulo de 2.
N   = 1024
Fs  = 1024 #1024 Hz es un poco extraño, por lo general viene del dominio de reloj de la computaora
Ts  = 1/Fs
fsig = 10


plt.close('all')

#Generamos un vector de tiempos. Permite mucha flexibilidad
#Tener en cuenta que la primera muestra es a tiempo 0 por eso va el N-1
#El tercer parametros es cada N muestras-
t = np.linspace(0.0,(N-1)/Fs,N)

#Señal senoidal
#s = np.zeros(N)
#s = np.sin(2*np.pi*fsig*t)

s = 0.5 + signal.square(2*np.pi*fsig*t)/2 



#2/N es el espectro normalizado, o sea el maximo va a llevar un '1'
spectrum = (2/N)*np.abs(sc.fft(s));

#La doble barra hace la division por enteros,para no tener un indice que no sea un valor con ","
half = spectrum[0:N//2]

frec = np.linspace(0,Fs/2,N//2)

plt.stem(frec,half)
plt.show()

plt.figure(2)
plt.plot(t,s)
plt.show()





