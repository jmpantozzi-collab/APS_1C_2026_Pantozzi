# -*- coding: utf-8 -*-
"""
Diseñar un generador de señales que utilizaremos en las primeras simulaciones que hagamos. La primer tarea consistirá en programar una función que genere señales senoidales y que permita parametrizar:

la amplitud máxima de la senoidal (volts): Vmax
su valor medio (volts): dc
la frecuencia (Hz): ff
la fase (radianes): ph
la cantidad de muestras digitalizada por el ADC (# muestras): nn 
la frecuencia de muestreo del ADC.: fs

@author: Jeremias Pantozzi
"""
#%% Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds

#%% funcion seno
def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
        tt= np.arange(nn)/fs
        xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
        return tt, xx

#%% main
vmax=1
dc=0
ff=101
ph=0
nn=100
fs=100

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

plt.plot(tt, xx)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal Senoidal Generada")
plt.grid(True)
plt.show()

