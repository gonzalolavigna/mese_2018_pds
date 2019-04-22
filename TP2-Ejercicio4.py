#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:12:40 2019

@author: glavigna
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
import scipy.io as sio
import scipy.fftpack as sc


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

def get_ECG_TP4_MAT():
    #Cargo el archivo con la informacion
    test = sio.loadmat('ECG_TP4.mat')
    ecg =test['ecg_lead']    
    hb_p1 =test['heartbeat_pattern1']   
    hb_p2 =test['heartbeat_pattern2']   
    qrs_detections =test['qrs_detections']   
    qrs_p1 =test['qrs_pattern1']   
    
    return ecg,hb_p1,hb_p2,qrs_detections,qrs_p1



#Cierro todos los graficos por default.
plt.close('all')

#Cargo el archivo con la informacion
test = sio.loadmat('ECG_TP4.mat')

#Hacemos una FFT de la señal correspondiente a un latido bueno
heartbeat_pattern_1 = test['heartbeat_pattern1'];
heartbeat_pattern_1 = heartbeat_pattern_1.flatten()
heartbeat_pattern_2 = test['heartbeat_pattern2'];
heartbeat_pattern_2 =  heartbeat_pattern_2.flatten()

#Al ECG completo 
ecg = test['ecg_lead']

#Frecuencia de sampleo es un 1Khz
fs = 1000
ts = 1/fs
N  = np.size(heartbeat_pattern_1)
tt = np.linspace(0, (N-1)*ts, N).flatten()


plt.figure(1)
line_hdls = plt.plot(tt,heartbeat_pattern_1)
plt.title('Pulso OK')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud en muestra')
plt.show()


(ff,half_fft) = simple_fft(heartbeat_pattern_1,fs,len(heartbeat_pattern_1))

plt.figure(2)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()


#Analisis de un pulso incorrecto
N  = np.size(heartbeat_pattern_2)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(3)
line_hdls = plt.plot(tt,heartbeat_pattern_2)
plt.title('Pulso NOK')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()


(ff,half_fft) = simple_fft(heartbeat_pattern_2,fs,len(heartbeat_pattern_2))

plt.figure(4)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()



##Analicemos un pedazo de baja frecuencia con la FFT
#Sacamos la parte donde hay pulsos con ruido como indica el ejercicio
zonas_con_interf_baja_frec = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )


zonas_sin_interf = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

ecg = ecg.flatten()

#Analisis de una seccion con ruido baja frecuencia

slice_baja_frec = zonas_con_interf_baja_frec[0].astype(int)
ecg_slice = ecg[slice_baja_frec[0]:slice_baja_frec[1]]

N  = np.size(ecg_slice)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(5)
line_hdls = plt.plot(tt,ecg_slice)
plt.title('SLICE BAJA FRECUENCIA')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [muestras]')
plt.show()

(ff,half_fft) = simple_fft(ecg_slice,fs,len(ecg_slice))

plt.figure(6)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()

#Analisis de una seccion sin ruido de baja frecuencia
slice_ok = zonas_sin_interf[0].astype(int)
ecg_slice = ecg[slice_ok[0]:slice_ok[1]]

N  = np.size(ecg_slice)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(7)
line_hdls = plt.plot(tt,ecg_slice)
plt.title('SLICE BAJA FRECUENCIA')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [muestras]')
plt.show()

(ff,half_fft) = simple_fft(ecg_slice,fs,len(ecg_slice))

plt.figure(8)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()



