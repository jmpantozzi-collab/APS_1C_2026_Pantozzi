# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:01:46 2026

@author: jerem
"""

#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from scipy.io.wavfile import write

#%%SEÑALES

ecg_one_lead = np.load('ecg_sin_ruido.npy')
fs_ecg = 1000

#cucaracha
fs_audio_cuca, wav_data_cuca = sio.wavfile.read('la cucaracha.wav')
#%%WELCH
#Agregar zero-padding para ver todo en resolucion espectral.

#Para calcular Bw tenemos que hacer acumulacion de potencia (miramos a la izquierda de los puntos) y luego establecemos cuanta
#potencia la asociamos al rudio, ejemplo 95% de potencia asociada a la señal, y un 5% al ruido (muy poco ruido).
#en pasabanda asumo que el ruido esta dividido en 2, 2.5% de un lado y 2.5% del otro por ejemplo, para limitar el inicio y el final
#pasa bajo es mas facil... el bw es hasta acumular el 95% y listo.
#Plantear el grafico para verlo

#Normalizacion, elegir un criterio, podemos por ejemplo que el area sea 1, normslizar segun el maximo, etc.

#Audio
nn=140000

# Definimos el tamaño del zero-padding. Generalmente es una potencia de 2 mayor a nperseg, 
# o simplemente un multiplicador. Aquí usaremos el doble del nn máximo para el padding general.
nfft_pad = nn * 2

#f_cuca, Pxx_den_cuca = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn,nfft=nfft_pad)
#f_cuca2, Pxx_den_cuca2 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/2,nfft=nfft_pad)
#f_cuca4, Pxx_den_cuca4 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/4,nfft=nfft_pad)
f_cuca8, Pxx_den_cuca8 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/7,nfft=nfft_pad)
#f_cuca10, Pxx_den_cuca10 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/10,nfft=nfft_pad)


umbral_db = -40

# Conversión a dB
#pxx_cuca_db = 10 * np.log10(Pxx_den_cuca)
#pxx_cuca2_db = 10 * np.log10(Pxx_den_cuca2)
#pxx_cuca4_db = 10 * np.log10(Pxx_den_cuca4)
pxx_cuca8_db = 10 * np.log10(Pxx_den_cuca8)
#pxx_cuca10_db = 10 * np.log10(Pxx_den_cuca10)

psd_filtrado_lineal = np.copy(Pxx_den_cuca8)
psd_filtrado_lineal[pxx_cuca8_db < umbral_db] = 0

potencia_acumulada = np.cumsum(psd_filtrado_lineal)
potencia_total = potencia_acumulada[-1]
potencia_acumulada_norm = potencia_acumulada / potencia_total

indice_2_5 = np.where(potencia_acumulada_norm >= 0.025)[0][0]
indice_95 = np.where(potencia_acumulada_norm >= 0.95)[0][0]
plt.figure(1, figsize=(12, 7))

# Usamos plt.plot en lugar de plt.semilogy
#plt.plot(f_cuca, pxx_cuca_db, color='red', label=f'nperseg = {nn}', linewidth=1)
#plt.plot(f_cuca2, pxx_cuca2_db, color='blue', label=f'nperseg = {nn//2}', linewidth=1)
#plt.plot(f_cuca4, pxx_cuca4_db, color='green', label=f'nperseg = {nn//4}', linewidth=1)
plt.plot(f_cuca8, pxx_cuca8_db, color='violet', label=f'nperseg = {nn//7}', linewidth=1)
#plt.plot(f_cuca10, pxx_cuca10_db, color='pink', label=f'nperseg = {nn//10}', linewidth=1)

#plt.semilogy(f_cuca, Pxx_den_cuca, color='red', label=f'nperseg = {nn}', linewidth=1)
#plt.semilogy(f_cuca2, Pxx_den_cuca2, color='blue', label=f'nperseg = {nn//2}', linewidth=1)
#plt.semilogy(f_cuca4, Pxx_den_cuca4, color='green', label=f'nperseg = {nn//4}', linewidth=1)
#plt.semilogy(f_cuca8, Pxx_den_cuca8, color='violet', label=f'nperseg = {nn//8}', linewidth=1)
#plt.semilogy(f_cuca10, Pxx_den_cuca10, color='pink', label=f'nperseg = {nn//10}', linewidth=1)
plt.title('Espectro de Potencia - La Cucaracha (con Zero-Padding)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend() # Añadido para mostrar las etiquetas
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()


#ecg
nn_ecg=30000
nfft_ecg_pad = nn_ecg * 2 # Zero-padding aplicado al ECG

#f_ecg, Pxx_den_ecg = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=nn_ecg,nfft=nfft_ecg_pad)
#f_ecg2, Pxx_den_ecg2 = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=nn_ecg/2,nfft=nfft_ecg_pad)
f_ecg4, Pxx_den_ecg4 = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=nn_ecg/4,nfft=nfft_ecg_pad)
#f_ecg8, Pxx_den_ecg8 = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=nn_ecg/8,nfft=nfft_ecg_pad)
#f_ecg10, Pxx_den_ecg10 = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=nn_ecg/10,nfft=nfft_ecg_pad)

Pxx_den_ecg4_norm=Pxx_den_ecg4/np.max(np.abs(Pxx_den_ecg4))
# Conversión a dB
#pxx_ecg_db = 10 * np.log10(Pxx_den_ecg)
#pxx_ecg2_db = 10 * np.log10(Pxx_den_ecg2)
pxx_ecg4_db = 10 * np.log10(Pxx_den_ecg4_norm)
#pxx_ecg8_db = 10 * np.log10(Pxx_den_ecg8)
#pxx_ecg10_db = 10 * np.log10(Pxx_den_ecg10)


plt.figure(2, figsize=(12, 7))
#plt.semilogy(f_ecg, Pxx_den_ecg, color='red', label=f'nperseg = {nn}', linewidth=1)
#plt.semilogy(f_ecg2, Pxx_den_ecg2, color='b', label=f'nperseg = {nn//2}', linewidth=1)
#plt.semilogy(f_ecg4, Pxx_den_ecg4, color='pink', label=f'nperseg = {nn//4}', linewidth=1)
#plt.semilogy(f_ecg8, Pxx_den_ecg8, color='violet', label=f'nperseg = {nn//8}', linewidth=1)
#plt.semilogy(f_ecg10, Pxx_den_ecg10, color='green', label=f'nperseg = {nn//10}', linewidth=1)
#plt.ylim([0.5e-3, 1])

#plt.plot(f_ecg, pxx_ecg_db, color='red', label=f'nperseg = {nn_ecg}', linewidth=1)
#plt.plot(f_ecg2, pxx_ecg2_db, color='b', label=f'nperseg = {nn_ecg//2}', linewidth=1)
plt.plot(f_ecg4, pxx_ecg4_db, color='pink', label=f'nperseg = {nn_ecg//4}', linewidth=1)
#plt.plot(f_ecg8, pxx_ecg8_db, color='violet', label=f'nperseg = {nn_ecg//8}', linewidth=1)
#plt.plot(f_ecg10, pxx_ecg10_db, color='green', label=f'nperseg = {nn_ecg//10}', linewidth=1)

plt.title('Espectro de Potencia - ECG (con Zero-Padding)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend() # Añadido para mostrar las etiquetas
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()