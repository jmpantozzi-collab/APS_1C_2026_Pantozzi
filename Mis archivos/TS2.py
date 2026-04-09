# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:13:24 2026

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
ff=2/1000
ph=0
nn=1000
fs=2

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

#%% Parametros de ADC
B = 4 #bits

Vfs = 2 #Volts

qq= Vfs / (2**B)

#%% funcion ruido segun dB pedido

Pr=(qq**2)/12
R= np.random.normal(0, np.sqrt(Pr), nn)


xxn=xx + R

plt.figure(1)
plt.clf()
plt.plot (xx, label ='S')
plt.plot(xxn, label = 'S+r')
plt.legend()

#%% fft


XX = np.fft.fft(xx)*1/nn #escalamos con 1/nn para ver W

XXmod= np.abs(XX)
XXphase= np.angle(XX)
XXmod_ad=(XXmod**2)#Ajustamos con factor de escala 2 porque tengo visualizado 1, por simetria la otra es igual

XXmod_dB = 10 * np.log10(XXmod_ad) 


#%% fft señal + ruido

XXr = np.fft.fft(xxn)*1/nn

XXrmod= np.abs(XXr)
XXrphase= np.angle(XXr)

plt.figure(3, figsize=(10, 8))
plt.clf()
plt.title = ('FFT de XXr')

XXrmod_dB = 10 * np.log10((XXrmod)**2)


#%% fft de cuantizacion

xxq= np.round(xxn/qq)*qq

plt.figure(1)
plt.clf()
plt.plot (tt,xx, label ='S', color='blue')
plt.plot(tt,xxn, label = 'S+r', color= 'orange')
plt.plot(tt,xxq, label= 'Sq', color='red')
plt.legend()

exxq= xxq-xxn

XXq = np.fft.fft(xxq)*1/nn

XXqmod= np.abs(XXq)
XXqphase= np.angle(XXq)
XXqmod_dB = 10 * np.log10((XXqmod)**2)



#%% piso analogico y digital

piso_analog = np.mean(XXrmod_dB) #piso analogico valor medio.
piso_digital= np.mean(XXqmod_dB) 

plt.figure(2)
plt.clf()
plt.title = ('FFT')
plt.plot (XXqmod_dB[:500], label = 'Sq-ADC out', color = 'red',linestyle='--')
plt.plot (XXmod_dB[:500], label = 'S', color = 'blue')
plt.plot (XXrmod_dB[:500], label = 'S+r', color = 'orange',linestyle='dotted')
plt.plot([piso_analog]*500, color='black', linestyle='dashed', label='piso analogico')
plt.plot([piso_digital]*500, color='violet', linestyle='dashed', label='piso digital')

plt.legend()
