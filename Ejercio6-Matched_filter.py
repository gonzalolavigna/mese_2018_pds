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

#Cierro todos los graficos por default.
plt.close('all')

#Definimos la fs que es transversal a todo y tambien al ts
fs = 1000
ts = 1/fs

ecg,hb_p1,hb_p2,qrs_detections,qrs_p1 = get_ECG_TP4_MAT()

#Hagamos una cascada de tecnica a la señal, primero aplicamos el filtro de mediana.
#Esta es la tecnica que habiamos implementado antes, me parece la mas exacta
ecg_median_filter = signal.medfilt(ecg,201)
ecg_median_filter = signal.medfilt(ecg_median_filter ,601)


ecg_hp = ecg - ecg_median_filter 
#Load low pass FIR
files = np.load('FIR_LOWPASS.npz')
coefficient_lp=files['ba.npy']

#Despues aplicamos el filtro pasa alto, utilizamos el FIR equirriple del TP4.
b_lp = coefficient_lp[0].flatten()
a_lp = coefficient_lp[1].flatten()

ecg_filtered = signal.filtfilt(b_lp,a_lp,ecg_hp );

#Ya tenemos la señal filtrada con lo cual las ploteamos para ver como queda 
zonas_sin_interf = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

#Ahora vamos a probar el Filtro en una zona sin ruido de baja frecuencia
slice_sin_baja_frec     =  zonas_sin_interf[0].astype(int)

plot_slice(slice_sin_baja_frec,ecg,ecg_hp ,ecg_filtered)
plt.title('SLICE SIN RUIDO DE BAJA FRECUENCIA')
plt.show()

#Hagamos la correlacion, esta es la señal con la que queremos correlacionar
N = np.size(qrs_p1)
tt = np.linspace(0,(N-1)*ts,N).flatten()
plt.figure()
plt.plot(tt,qrs_p1,'r')
plt.plot(tt,np.flip(qrs_p1),'b')

#Hacemos el filtro inverso
ecg_inv_filter = signal.lfilter(np.flip(hb_p1),[1],ecg_filtered)

#corr = signal.correlate(ecg_filtered,qrs_p1, mode = 'same')
#invertamos la señal patron qrs_p1




