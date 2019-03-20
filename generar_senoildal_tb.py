#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 05:31:07 2019

@author: glavigna
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

senoidal = generador_senoidal(fs = 1000, f0 = 1, N = 2000, a0 = 2, p0 = np.pi);

plt.figure(1)
line_hdls = plt.plot(senoidal[0], senoidal[1])
plt.title('Señal: ' + 'Senoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
       
plt.show()