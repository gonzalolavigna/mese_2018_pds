#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 04:53:29 2019

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
    ecg =test['ecg_lead'].flatten()    
    hb_p1 =test['heartbeat_pattern1'].flatten()   
    hb_p2 =test['heartbeat_pattern2'].flatten()   
    qrs_detections =test['qrs_detections'].flatten()   
    qrs_p1 =test['qrs_pattern1'].flatten()   
    
    return ecg,hb_p1,hb_p2,qrs_detections,qrs_p1

#Cierro todos los graficos por default.
plt.close('all')


ecg,hb_p1,hb_p2,qrs_detections,qrs_p1 = get_ECG_TP4_MAT()

#Load high pass FIR
files = np.load('FIR_HIGHBAND.npz')
coefficient=files['ba.npy']

b = coefficient[0].flatten()
a = coefficient[1].flatten()

filtered_ecg_1 = signal.filtfilt(b,a,ecg)


#Frecuencia de sampleo es 1Khz
fs = 1000;
ts = 1/fs
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

#Analisis de una seccion con ruido baja frecuencia
slice_baja_frec = zonas_con_interf_baja_frec[0].astype(int)
ecg_slice_1 = ecg[slice_baja_frec[0]:slice_baja_frec[1]]
filtered_ecg_slice_1 = filtered_ecg_1[slice_baja_frec[0]:slice_baja_frec[1]]

N  = np.size(ecg_slice_1)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(1)
line_hdls = plt.plot(tt,ecg_slice_1,'b')
line_hdls_2 = plt.plot(tt,filtered_ecg_slice_1,'r')
plt.title('SLICE BAJA FRECUENCIA')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [muestras]')
plt.show()

N  = np.size(hb_p1)
tt = np.linspace(0, (N-1)*ts, N).flatten()
plt.figure(2)
line_hdls = plt.plot(tt,hb_p1)
plt.title('Pulso OK')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud en muestra')
plt.show()

#load low pass FIR
#Load low band pass FIR
files = np.load('FIR_LOWPASS.npz')
coefficient=files['ba.npy']

b = coefficient[0].flatten()
a = coefficient[1].flatten()

#Volvemos a pasar la señal filtrada por el FIR
filtered_ecg_2 = signal.filtfilt(b,a,filtered_ecg_1)

#Analisis de una seccion con ruido baja frecuencia
slice_baja_frec = zonas_con_interf_baja_frec[0].astype(int)
ecg_slice_2 = ecg[slice_baja_frec[0]:slice_baja_frec[1]]
filtered_ecg_slice_2 = filtered_ecg_2[slice_baja_frec[0]:slice_baja_frec[1]]

N  = np.size(ecg_slice_2)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(3)
line_hdls = plt.plot(tt,ecg_slice_2,'b')
line_hdls_2 = plt.plot(tt,filtered_ecg_slice_1,'g')
line_hdls_3 = plt.plot(tt,filtered_ecg_slice_2,'r')
plt.title('SLICE BAJA FRECUENCIA')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [muestras]')
plt.show()


#Ahora vamos a probar el Filtro FIR en una zona sin ruido de baja frecuencia
slice_sin_baja_frec     =  zonas_sin_interf[0].astype(int)
ecg_sin_baja_sclice             = ecg[slice_sin_baja_frec[0]:slice_sin_baja_frec[1]]
filtered_ecg_sin_baja_slice_1   = filtered_ecg_1[slice_sin_baja_frec[0]:slice_sin_baja_frec[1]]
filtered_ecg_sin_baja_slice_2   = filtered_ecg_2[slice_sin_baja_frec[0]:slice_sin_baja_frec[1]]

N  = np.size(ecg_sin_baja_sclice)
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(4)
line_hdls = plt.plot(tt,ecg_sin_baja_sclice,'b')
line_hdls_2 = plt.plot(tt,filtered_ecg_sin_baja_slice_1,'g')
line_hdls_3 = plt.plot(tt,filtered_ecg_sin_baja_slice_2,'r')
plt.title('SLICE SIN RUIDO BAJA FRECUENCIA')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [muestras]')
plt.show()


(ff,half_fft) = simple_fft(ecg_sin_baja_sclice,fs,len(ecg_sin_baja_sclice))

plt.figure(5)
plt.stem(ff,half_fft)
plt.title('Espectro de la señal haciendo FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud Normalizada')
plt.show()