#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:57:17 2019

@author: glavigna
"""

#Cargar un archivo NPZ en 
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
import scipy.io as sio
import scipy.fftpack as sc

def get_ECG_TP4_MAT():
    #Cargo el archivo con la informacion
    test = sio.loadmat('ECG_TP4.mat')
    ecg =test['ecg_lead'].flatten()    
    hb_p1 =test['heartbeat_pattern1'].flatten()   
    hb_p2 =test['heartbeat_pattern2'].flatten()   
    qrs_detections =test['qrs_detections'].flatten()   
    qrs_p1 =test['qrs_pattern1'].flatten()   
    
    return ecg,hb_p1,hb_p2,qrs_detections,qrs_p1

def plot_slice (slice_ecg,ecg_original,ecg_hp,ecg_hp_lp):
    ecg_slice       = ecg[slice_ecg[0]:slice_ecg[1]]
    ecg_hp_slice    = ecg_hp[slice_ecg[0]:slice_ecg[1]]
    ecg_hp_lp_slice = ecg_hp_lp[slice_ecg[0]:slice_ecg[1]]
    
    #Calculamos el vector para los tiempos
    N  = np.size(ecg_slice);
    tt = np.linspace(0,(N-1)*ts,N).flatten()
    plt.figure()
    line_hdls_1 = plt.plot(tt,ecg_slice,'b')
    line_hdls_2 = plt.plot(tt,ecg_hp_slice,'g')
    line_hdls_3 = plt.plot(tt,ecg_hp_lp_slice,'r')
    
    plt.xlabel('tiempo [segundos]')
    plt.ylabel('Amplitud [muestras]')
    #plt.show()


ecg,hb_p1,hb_p2,qrs_detections,qrs_p1 = get_ECG_TP4_MAT()

#Definimos la fs que es transversal a todo y tambien al ts
fs = 1000
ts = 1/fs

zonas_con_interf_baja_frec = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )


zonas_sin_interf = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

#Hacemos una señal steam que sirva para graficar la parte correspondientes

array_filled = np.zeros(np.size(qrs_detections))
for i in range(np.size(qrs_detections)):
    array_filled[i] = ecg[qrs_detections[i]];

N = np.size(ecg[0:1999]);
tt = np.linspace(0,(N-1)*ts,N).flatten()

#plt.figure()
#line_hdls_1 = plt.plot(tt,ecg[0:1999],'b')
#line_hdls_2 = plt.stem(tt,array_filled[0:1999])
    

#Hacemos nuestro filtros estimando la linea de base con 100 muestras antes
delay = 150
base_line_y = np.zeros(np.size(qrs_detections)+1);
base_line_x = np.zeros(np.size(qrs_detections)+1);
for i in range (np.size(qrs_detections)):
    base_line_y[i] = ecg[qrs_detections[i]-delay]
    base_line_x[i] = qrs_detections[i]-delay
    
base_line_x[-1] = np.size(ecg)
base_line_y[-1] = ecg[-1]

base_line_x = base_line_x.astype(int)

cs = CubicSpline(base_line_x, base_line_y)

plt.figure(1)

plt.plot(qrs_detections,array_filled,'x',label = 'data')
plt.plot(base_line_x,base_line_y,'o',label = 'data')
plt.plot(ecg,'r')
#plt.plot(cs).

plt.figure(2)


N = np.size(ecg);
#tt = np.linspace(0,(N-1)*ts,N).flatten()
xs = np.arange(0,N,1)

ys = cs(xs)
plt.plot(ys)

#Analisis de una seccion con ruido baja frecuencia
slice_baja_frec = zonas_con_interf_baja_frec[0].astype(int)

plot_slice(slice_baja_frec,ecg,ys ,ecg-ys)


slice_sin_baja_frec     =  zonas_sin_interf[0].astype(int)
plot_slice(slice_sin_baja_frec,ecg,ys ,ecg-ys)


    
    





