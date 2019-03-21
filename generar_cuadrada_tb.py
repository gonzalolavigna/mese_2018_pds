#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:14:24 2019

@author: glavigna
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from   scipy import signal

def generar_cuadrada (fs, f0, N, a0=1, duty = 0.5):

    #Chequeo el duty
    if(duty > 1.0 or duty < 0.0):
        raise Exception('duty debe estar en 0.0 y 1.0. El valor de duty fue: {}'.format(duty))    
    
    ts = 1/fs # tiempo de muestreo        
    #Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
    #Flatten convierte a un array de 1 dimensión.
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Concatenación de matrices:
    # guardaremos las señales creadas al ir poblando la siguiente matriz vacía
    yy = np.array([], dtype=np.float).reshape(N,0)
    
    #Genero la senoidal
    #signal = a0 * np.sin(2 * np.pi * f0 * tt + p0);
    yy  = a0 * signal.square(2 * np.pi * f0 * tt , duty)
   
    return tt,yy

fs = 1000
f0 = 1.5
N  = 1000
a0 = 0.25
duty = 0.5


[tt,yy] = generar_cuadrada(fs,f0,N,a0,duty)

plt.figure(1)
line_hdls = plt.plot(tt, yy)
plt.title('Señal: ' + 'Cuadrada')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')