#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:57:17 2019

@author: glavigna
"""

#Cargar un archivo NPZ en 
import numpy as np
from scipy import signal
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

#Analisis de una seccion con ruido baja frecuencia
slice_baja_frec = zonas_con_interf_baja_frec[0].astype(int)


ecg_median_filter = signal.medfilt(ecg,201)
ecg_median_filter = signal.medfilt(ecg_median_filter ,601)

plot_slice(slice_baja_frec,ecg,ecg_median_filter ,ecg-ecg_median_filter)
plt.title('SLICE CON RUIDO DE BAJA FRECUENCIA')
plt.show()

#Ahora vamos a probar el Filtro en una zona sin ruido de baja frecuencia
slice_sin_baja_frec     =  zonas_sin_interf[0].astype(int)
plot_slice(slice_sin_baja_frec,ecg,ecg_median_filter ,ecg-ecg_median_filter)
plt.title('SLICE SIN RUIDO DE BAJA FRECUENCIA')
plt.show()




