# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:05:58 2026

DFT (FFT en compu) de x[n]=4+3sin[pi/2 * n]
8 muestras 

graficar modulo y fase
@author: jerem
"""

#%% Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal as sig

#%% funcion seno
def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
        tt= np.arange(nn)/fs
        xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
        return tt, xx
    
#%%Señal

vmax=3
dc=4
ff=1/4
ph=0
nn=1000
fs=1000

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

a=2 * np.pi * ff 