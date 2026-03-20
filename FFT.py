# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:45:14 2026

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

#%% ejecucion
vmax=1
dc=0
ff=1
ph=0
nn=1000
fs=1000

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

#%% fft
XX = np.fft.fft(xx)

XXmod= np.abs(XX)
XXphase= np.angle(XX)

plt.figure(1, figsize=(10, 8))
plt.clf()
plt.title = ('FFT de XX')
# --- Primer panel: modulo---
ax1 = plt.subplot(2, 1, 1)
plt.plot (XXmod, label = 'modulo', color = 'red')
plt.legend()

# --- Psegundo panel: fase---
ax2 = plt.subplot(2, 1, 2)
plt.plot(XXphase, label ='fase', color= 'blue')
plt.legend()
plt.tight_layout()
plt.show()

