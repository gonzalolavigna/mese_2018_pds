#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:00:16 2019

@author: glavigna
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from   scipy import signal


#Copia del generador de senoidal
def generador_senoidal (fs, f0, N, a0=1, p0=0):
    
    ts = 1/fs # tiempo de muestreo    
    
    #Genero el espacio para poder tener el espacio temporal que va de 0 a N-1
    #Flatten convierte a un array de 1 dimensión.
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Concatenación de matrices:
    # guardaremos las señales creadas al ir poblando la siguiente matriz vacía
    yy = np.array([], dtype=np.float).reshape(N,0)
    
    #Genero la senoidal
    yy = a0 * np.sin(2 * np.pi * f0 * tt + p0);
   
    return tt,yy

def cuantizar_senial(yy,N,Q):
    
    #yy_qq = np.array([], dtype=np.float).reshape(N,0)
    
    yy_qq = np.zeros(N)
    for k in range(N):
        yy_qq[k] = Q * np.round(yy[k]/Q)
    
    return yy_qq

#Frecuencia de oversampling
fos = 10000
f0  = 2
N   = 10000
a0  = 0.5
p0  = 0

[tt,yy] = generador_senoidal(fos,f0,N,a0,p0)

plt.figure(1)
line_hdls = plt.plot(tt, yy)
plt.title('Señal: ' + 'Senoidal Oversampling')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')

plt.show()

varianza = np.var(yy,dtype = np.float64)
print('Varianza: {}'.format(varianza))

#Genero una señal de ruido gaussianda con una varianza igual a la señal /10
noise = np.random.normal(0,np.sqrt(varianza/10),np.size(tt))

yy_nn = yy + noise;

plt.figure(1)
line_hdls = plt.plot(tt, yy_nn)
plt.title('Señal: ' + 'Senoidal + Ruido')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()

#Resampleo la señal a una frecuencia menor
fs = 1000
scale_factor = np.int(np.round(fos/fs))
#numero de muestras que quedan despues del sampleo
Nrs = np.int(N / scale_factor)

#Resampleo las señales reduciendo la frecuencia de oversampling
yy_nn_rs = yy_nn[1:yy_nn.size:scale_factor]
tt_rs = tt[1:tt.size:scale_factor]

plt.figure(1)
line_hdls = plt.plot(tt_rs, yy_nn_rs)
plt.title('Señal: ' + 'Senoidal + Ruido + Resampleada con un factor: {}'.format(scale_factor))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()

#Numero de bits que se desea hacer la adquisicion
nbits = 1
#VFS
vfs = 1
#Paso de la cuantizacion
Q = (1/2**(nbits-1))*vfs
print('Q step {}'.format(Q))

yy_qq = cuantizar_senial(yy_nn_rs,Nrs,Q)

plt.figure(1)
line_hdls = plt.plot(tt_rs, yy_qq)
plt.title('Señal: ' + 'Senoidal + Ruido + Resampleada con un factor: {} + Samplea a {} bits'.format(scale_factor,nbits))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()

unique_values = np.unique(yy_qq);
print("Valores unicos:{}".format(unique_values.size))

