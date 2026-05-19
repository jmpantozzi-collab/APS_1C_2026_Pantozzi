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

#%% funciones
def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
        tt= np.arange(nn)/fs
        xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
        return tt, xx
    
def calcular_piso_ruido(espectro_potencia):
    # Buscamos el índice donde está el pico (la señal)
    idx_senal = np.argmax(espectro_potencia)
    
    # Creamos una copia para no romper el espectro original
    solo_ruido = espectro_potencia.copy()
    
    # "Borramos" el pico de la señal (y sus vecinos por si hay leakage)
    # Seteamos a NaN para que np.nanmean lo ignore
    solo_ruido[idx_senal] = np.nan
    if idx_senal > 0: solo_ruido[idx_senal-1] = np.nan
    if idx_senal < len(solo_ruido)-1: solo_ruido[idx_senal+1] = np.nan
    
    # Promediamos solo lo que quedó (el ruido puro)
    return np.nanmean(solo_ruido)

#%% ejecucion
vmax=np.sqrt(2)
dc=0
ff=1
ph=0
nn=1000
fs=1000

tt,xx=mi_funcion_sen(vmax,dc,ff,ph,nn,fs)

#%% Parametros de ADC
B = 4 #bits

Vfs = 2 #Volts

qq= Vfs / (2**B)

#%% funcion ruido segun dB pedido
kn=1
Pr=kn*(qq**2)/12
R= np.random.normal(0, np.sqrt(Pr), nn)


xxn=xx + R

#%% fft
# Tomamos solo la mitad (frecuencias positivas)
half_nn = nn // 2

XX = np.fft.fft(xx)*1/nn #escalamos con 1/nn para ver W

XX_half = XX[:half_nn]
# Multiplicamos por 2 para recuperar la amplitud de la sinusoide
# (porque la FFT reparte la energía en frecuencias + y -)
XXmod= np.abs(XX_half)*2
# Referencia de potencia máxima de la señal pura, Normalizacion por maximo
ref_potencia = np.max(XXmod**2)

XXmod_dB = 10 * np.log10(XXmod**2 / ref_potencia)

#%% fft señal + ruido

XXr = np.fft.fft(xxn)*1/nn

XXrmod= np.abs(XXr[:half_nn]) * 2


XXrmod_dB = 10 * np.log10(XXrmod**2 / ref_potencia)


#%% fft de cuantizacion

xxq= np.round(xxn/qq)*qq

plt.figure(1, figsize=(10, 6))
plt.clf()
plt.plot (tt,xx, label ='S', color='blue')
plt.plot(tt,xxn, label = 'S+r', color= 'orange')
plt.plot(tt,xxq, label= 'Sq', color='red')
plt.legend()

exxq= xxq-xxn

XXq = np.fft.fft(xxq)*1/nn

XXqmod = np.abs(XXq[:half_nn]) * 2

XXqmod_dB = 10 * np.log10(XXqmod**2 / ref_potencia)



#%% piso analogico y digital

# Calculamos la potencia total de la señal de error (ruido) en el tiempo
# Esto es universal: no importa la forma del ruido ni donde esté la señal.
potencia_ruido_analog_total = np.var(R) 
potencia_ruido_digital_total = np.var(exxq) # exxq es el error de cuantización

#  La potencia que ves en CADA BIN del espectro es la potencia total 
# dividida por la cantidad de bins (nn/2)
potencia_por_bin_analog = potencia_ruido_analog_total / (nn/2)
potencia_por_bin_digital = potencia_ruido_digital_total / (nn/2)

# 3. Referencia de potencia de la señal (P = A^2 / 2)
# Usamos la amplitud que recuperamos en la FFT (XXmod tiene el factor 2)
P_senal = (np.max(XXmod)**2) / 2

# 4. Calculamos los niveles en dB
piso_analog = 10 * np.log10(potencia_por_bin_analog / P_senal)
piso_digital = 10 * np.log10(potencia_por_bin_digital / P_senal)

plt.figure(2, figsize=(10, 6))
plt.clf()
plt.title(r"Señal muestreada por un ADC de 4 bits – $\pm V_R = 2.0\ \text{V}$ – $q = %.3f\ \text{V}$" % qq, fontsize=14)
plt.plot(XXqmod_dB, label='Sq-ADC out', color='red', linestyle='--', alpha=0.7)
plt.plot(XXmod_dB, label='S', color='skyblue', linewidth=2)
plt.plot(XXrmod_dB, label='S+r', color='orange', linestyle='dotted')

# Líneas de piso corregidas
plt.axhline(y=piso_analog, color='black', linestyle='dashed', label=f'piso analógico = {piso_analog:.2f} dB')
plt.axhline(y=piso_digital, color='blue', linestyle='dashed', label=f'piso digital = {piso_digital:.2f} dB')

plt.ylim([piso_digital - 60, 10]) # Ajusta el zoom vertical
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()

#%% HIstograma

plt.figure(3, figsize=(10, 8))
plt.clf()
plt.title(r"Ruido de cuantizacion para 4 bits – $\pm V_R = 2.0\ \text{V}$ – $q = %.3f\ \text{V}$" % qq, fontsize=14)

# Bins alineados exactamente con el rango teórico
nbins = 10
bins = np.linspace(-qq/2, qq/2, nbins+1)

counts, bins_edges, _ = plt.hist(
    exxq,
    bins=bins,
    density=True,
    color='skyblue',
    label='Histograma real'
)

# Límites teóricos
plt.axvline(-qq/2, color='red', linestyle='--', linewidth=2, label='±q/2 (Límites)')
plt.axvline(qq/2, color='red', linestyle='--', linewidth=2)

# Línea teórica uniforme
plt.hlines(1/qq, -qq/2, qq/2, color='orange', linewidth=3, label='Distribución teórica (1/q)')

plt.xlabel('Amplitud del error (Volts)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(axis='y', alpha=0.75)
