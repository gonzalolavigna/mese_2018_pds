#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:45:09 2019

@author: glavigna
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from   scipy import signal

def generar_triangular (fs, f0, N, a0=1, sym = 0.5):
    #Chequeo el duty
    if(sym > 1.0 or sym < 0.0):
        raise Exception('Simetria debe estar en 0.0 y 1.0. El valor de duty fue: {}'.format(sym))    
    
    ts = 1/fs # tiempo de muestreo        
    #Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
    #Flatten convierte a un array de 1 dimensión.
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Concatenación de matrices:
    # guardaremos las señales creadas al ir poblando la siguiente matriz vacía
    yy = np.array([], dtype=np.float).reshape(N,0)
    
    yy = signal.sawtooth(2*np.pi*f0*tt,sym)
    
    
    return tt,yy


fs = 1000
f0 = 1
N  = 1000
a0 = 0.25
sym = 0.7


[tt,yy] = generar_triangular(fs,f0,N,a0,sym)

plt.figure(1)
line_hdls = plt.plot(tt, yy)
plt.title('Señal: ' + 'Triangular')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')