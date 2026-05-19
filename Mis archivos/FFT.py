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
vmax=np.sqrt(2)
dc=0
ff=1
ph=0
nn=1000
fs=1000

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

#%% funcion ruido segun dB pedido

Px=np.var(xx)
SNR=50
Pr=10**(-SNR/10)
R= np.random.normal(0, np.sqrt(Pr), 1000)

xxn=xx + R

#%% fft


XX = np.fft.fft(xx)

XXmod= np.abs(XX)
XXphase= np.angle(XX)

plt.figure(1, figsize=(10, 8))
plt.clf()
plt.title = ('FFT de XX')
# --- Primer panel: modulo---
ax1 = plt.subplot(2, 1, 1)
plt.plot (XXmod[:500], label = 'modulo', color = 'red')
plt.legend()

# --- Psegundo panel: fase---
ax2 = plt.subplot(2, 1, 2)
plt.plot(XXphase[:500], label ='fase', color= 'blue')
plt.legend()
plt.tight_layout()
plt.show()


#%% fft señal + ruido

XXr = np.fft.fft(xxn)

XXrmod= np.abs(XXr)
XXrphase= np.angle(XXr)

plt.figure(2, figsize=(10, 8))
plt.clf()
plt.title = ('FFT de XXr')

XXrmod_dB = 20 * np.log10(XXrmod)

# --- Primer panel: modulo---
ax1 = plt.subplot(2, 1, 1)
plt.plot (XXrmod_dB[:500], label = 'modulo', color = 'red')
plt.legend()

# --- Psegundo panel: fase---
ax2 = plt.subplot(2, 1, 2)
plt.plot(XXrphase[:500], label ='fase', color= 'blue')
plt.legend()
plt.tight_layout()
plt.show()

#%% fft de cuantizacion

B = 3 #bits

Vfs = 3 #Volts

qq= Vfs / (2**B)

xxq= np.round(xxn/qq)*qq



exxq= xxq-xxn

XXq = np.fft.fft(xxq)

XXqmod= np.abs(XXq)
XXqphase= np.angle(XXq)

XXqmod_dB = 20 * np.log10(XXqmod)


plt.figure(3, figsize=(10, 8))
plt.clf()
plt.title = ('FFT de XXq')

# --- Primer panel: modulo---
ax1 = plt.subplot(2, 1, 1)
plt.plot (XXqmod_dB[:500], label = 'modulo', color = 'red')
plt.legend()

# --- Psegundo panel: fase---
ax2 = plt.subplot(2, 1, 2)
plt.plot(XXqphase[:500], label ='fase', color= 'blue')
plt.legend()
plt.tight_layout()
plt.show()