# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:05:58 2026

DFT (FFT en compu) de x[n]=4+3sin[pi/2 * n]
8 muestras 

Se ve que al tener 8  muestras el peridodo 2pi de la DTFT se divide en 2pi/8, por lo que mi frecuencia
debe caer en uno de esos puntos para que no me afecte a la lectura, lo que ocurre en XX2 (Desparramo espectral). 
Si quiero ver una frecuencia determinada debo tomar como muestras N talque 2pi/N=Frecuencia. 



graficar modulo y fase
@author: jerem
"""

#%% Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal as sig

#%%Señal
nn=8
n=np.arange(nn)
xx=4+3*np.sin((np.pi/2)*n)

xx2=4+3*np.sin(np.pi*(1/3)*n)

#%% fft


XX = np.fft.fft(xx)*1/nn #Normalizamos con 1/nn para ver W

XX2 = np.fft.fft(xx2)

XXmod= np.abs(XX)
XXphase= np.angle(XX)
XXmod_en=(XXmod**2) #Ajustamos con factor de escala 2 porque tengo visualizado 1, por simetria la otra es igual
 #Escalon para evitar valores 0 que rompen el log



XX2mod= np.abs(XX2)
XX2phase= np.angle(XX2)
XX2mod_en=(XX2mod**2)/(nn**2) #La potencia es (1/(N**2))*|X[k]|**2 si normalizo en la FFT este 1/n**2 aparece solo al hacer modulo al cuadro de la fft normalizada (como me paso en caso 1)

plt.figure(1, figsize=(10, 8))
plt.clf()
plt.suptitle('FFT de XX')

# --- Primer panel: módulo ---
ax1 = plt.subplot(2, 1, 1)
plt.stem(XXmod, linefmt='r-', markerfmt='ro', basefmt='k-', label='módulo de señal')
plt.legend()

# --- Segundo panel: fase ---
ax2 = plt.subplot(2, 1, 2)
plt.stem(XXphase, linefmt='b-', markerfmt='bo', basefmt='k-', label='fase')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(2, figsize=(10, 8))
plt.clf()
plt.suptitle('FFT Potencia de XX')

# --- Primer panel: módulo ---
ax1 = plt.subplot(2, 1, 1)
plt.stem(XXmod_en, linefmt='r-', markerfmt='ro', basefmt='k-', label='módulo al cuadrado de la señal')
plt.legend()

# --- Segundo panel: fase ---
ax2 = plt.subplot(2, 1, 2)
plt.stem(XXphase, linefmt='b-', markerfmt='bo', basefmt='k-', label='fase')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(3, figsize=(10, 8))
plt.clf()
plt.suptitle('FFT de XX2')

# --- Primer panel: módulo ---
ax1 = plt.subplot(3, 1, 1)
plt.stem(XX2mod, linefmt='r-', markerfmt='ro', basefmt='k-', label='módulo de señal')
plt.legend()

# --- Segundo panel: fase ---
ax2 = plt.subplot(3, 1, 2)
plt.stem(XX2phase, linefmt='b-', markerfmt='bo', basefmt='k-', label='fase')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(4, figsize=(10, 8))
plt.clf()
plt.suptitle('FFT Potecia de XX2')

# --- Primer panel: módulo ---
ax1 = plt.subplot(4, 1, 1)
plt.stem(XX2mod_en, linefmt='r-', markerfmt='ro', basefmt='k-', label='módulo al cuadrado de la señal')
plt.legend()

# --- Segundo panel: fase ---
ax2 = plt.subplot(4, 1, 2)
plt.stem(XX2phase, linefmt='b-', markerfmt='bo', basefmt='k-', label='fase')
plt.legend()

plt.tight_layout()
plt.show()


