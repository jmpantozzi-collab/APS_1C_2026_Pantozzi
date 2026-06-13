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
####################################
##            ECG                 ##
####################################
ecg_one_lead = np.load('ecg_sin_ruido.npy')
fs_ecg = 1000

####################################
##          PPG sin ruido         ##
####################################

fs_ppg = 400 # Hz
ppg = np.load('ppg_sin_ruido.npy')


####################################
##          CUCARACHA             ##
####################################
fs_audio_cuca, wav_data_cuca = sio.wavfile.read('la cucaracha.wav')
#%%WELCH


#Agregar zero-padding para ver todo en resolucion espectral.

#Para calcular Bw tenemos que hacer acumulacion de potencia (miramos a la izquierda de los puntos) y luego establecemos cuanta
#potencia la asociamos al rudio, ejemplo 95% de potencia asociada a la señal, y un 5% al ruido (muy poco ruido).
#en pasabanda asumo que el ruido esta dividido en 2, 2.5% de un lado y 2.5% del otro por ejemplo, para limitar el inicio y el final
#pasa bajo es mas facil... el bw es hasta acumular el 95% y listo.
#Plantear el grafico para verlo

#Normalizacion, elegir un criterio, podemos por ejemplo que el area sea 1, normslizar segun el maximo, etc.


#%%CUCHARCHA 

nn=140000

# Definimos el tamaño del zero-padding. Generalmente es una potencia de 2 mayor a nperseg, 
# o simplemente un multiplicador. 
nfft_pad = nn * 2

#f_cuca, Pxx_den_cuca = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn,nfft=nfft_pad)
#f_cuca2, Pxx_den_cuca2 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/2,nfft=nfft_pad)
#f_cuca4, Pxx_den_cuca4 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/4,nfft=nfft_pad)
f_cuca8, Pxx_den_cuca8 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/7,nfft=nfft_pad)
#f_cuca10, Pxx_den_cuca10 = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=nn/10,nfft=nfft_pad)

#Normalizacion
Pxx_den_cuca8_norm = Pxx_den_cuca8 / np.max(np.abs(Pxx_den_cuca8))

# Conversión a dB
#pxx_cuca_db = 10 * np.log10(Pxx_den_cuca)
#pxx_cuca2_db = 10 * np.log10(Pxx_den_cuca2)
#pxx_cuca4_db = 10 * np.log10(Pxx_den_cuca4)
pxx_cuca8_db_norm = 10 * np.log10(Pxx_den_cuca8_norm)
#pxx_cuca10_db = 10 * np.log10(Pxx_den_cuca10)



#Calculo de bw

#Filtrado de Piso de ruido (en lineal, pero usando la máscara normalizada)
umbral_db_audio = -80
psd_filtrado_lineal = np.copy(Pxx_den_cuca8)
psd_filtrado_lineal[pxx_cuca8_db_norm < umbral_db_audio] = 0

potencia_acumulada = np.cumsum(psd_filtrado_lineal)
potencia_total = potencia_acumulada[-1]
potencia_acumulada_norm = potencia_acumulada / potencia_total

indice_inferior = np.where(potencia_acumulada_norm >= 0.025)[0][0]
indice_superior = np.where(potencia_acumulada_norm >= 0.975)[0][0]

# Y para sacar las frecuencias (asumiendo que tenés la variable f_cuca8):
f_cuca_inf = f_cuca8[indice_inferior]
f_cuca_sup = f_cuca8[indice_superior]
bw_efectivo = f_cuca_sup - f_cuca_inf

print(f"Banda pasante: de {f_cuca_inf:.2f} Hz a {f_cuca_sup:.2f} Hz")
print(f"Ancho de Banda (BW) al 95%: {bw_efectivo:.2f} Hz")
plt.figure(1, figsize=(12, 7))  


#plt.plot(f_cuca, pxx_cuca_db, color='red', label=f'nperseg = {nn}', linewidth=1)
#plt.plot(f_cuca2, pxx_cuca2_db, color='blue', label=f'nperseg = {nn//2}', linewidth=1)
#plt.plot(f_cuca4, pxx_cuca4_db, color='green', label=f'nperseg = {nn//4}', linewidth=1)
plt.plot(f_cuca8, pxx_cuca8_db_norm, color='violet', label=f'nperseg = {nn//7}', linewidth=1)
#plt.plot(f_cuca10, pxx_cuca10_db, color='pink', label=f'nperseg = {nn//10}', linewidth=1)


plt.axvline(x=f_cuca_inf, color='darkviolet', linestyle='--', linewidth=1.5, label=f"F_inferior (2.5% BW): {f_cuca_inf:.2f} Hz")
plt.axvline(x=f_cuca_sup, color='darkviolet', linestyle='-', linewidth=1.5, label=f"F_superior (97.5% BW): {f_cuca_sup:.2f} Hz")
plt.title(f"Espectro de Potencia - La Cucaracha (con Zero-Padding) Bw: {bw_efectivo:.2f} Hz")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend() # Añadido para mostrar las etiquetas
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()


#%%ECG
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


#Calculo de Bw
umbral_db_ecg = -80
psd_ecg_filtrado = np.copy(Pxx_den_ecg4)
psd_ecg_filtrado[pxx_ecg4_db < umbral_db_ecg] = 0
potencia_acumulada_ecg = np.cumsum(psd_ecg_filtrado)
potencia_total_ecg = potencia_acumulada_ecg[-1]
potencia_acumulada_ecg_norm = potencia_acumulada_ecg / potencia_total_ecg

#Encontrar el índice donde se alcanza el 95% de la energía
indice_bw_ecg = np.where(potencia_acumulada_ecg_norm >= 0.95)[0][0]

#Buscar a qué frecuencia corresponde ese índice
bw_ecg = f_ecg4[indice_bw_ecg]

plt.figure(2, figsize=(12, 7))

#plt.plot(f_ecg, pxx_ecg_db, color='red', label=f'nperseg = {nn_ecg}', linewidth=1)
#plt.plot(f_ecg2, pxx_ecg2_db, color='b', label=f'nperseg = {nn_ecg//2}', linewidth=1)
plt.plot(f_ecg4, pxx_ecg4_db, color='pink', label=f'nperseg = {nn_ecg//4}', linewidth=1)
#plt.plot(f_ecg8, pxx_ecg8_db, color='violet', label=f'nperseg = {nn_ecg//8}', linewidth=1)
#plt.plot(f_ecg10, pxx_ecg10_db, color='green', label=f'nperseg = {nn_ecg//10}', linewidth=1)


plt.axvline(bw_ecg, color='hotpink', linestyle='-', linewidth=1.5, label=f"F (95% BW): {bw_ecg: .2f} Hz")
plt.title(f"Espectro de Potencia - ECG (con Zero-Padding) Bw: {bw_ecg: .2f} Hz")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend() # Añadido para mostrar las etiquetas
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

#%%PPG sin ruido

nn_ppg=50000
# Definimos el tamaño del zero-padding. Generalmente es una potencia de 2 mayor a nperseg, 
# o simplemente un multiplicador. 
nfft_pad_PPG = nn_ppg * 2

f_PPG, Pxx_den_ppg = sig.welch(ppg, fs_ppg,window='hamming',scaling='spectrum',nperseg=nn_ppg,nfft=nfft_pad_PPG)
f_PPG2, Pxx_den_ppg2 = sig.welch(ppg, fs_ppg,window='hamming',scaling='spectrum',nperseg=nn_ppg/2,nfft=nfft_pad_PPG)
f_PPG4, Pxx_den_ppg4 = sig.welch(ppg, fs_ppg,window='hamming',scaling='spectrum',nperseg=nn_ppg/4,nfft=nfft_pad_PPG)
f_PPG8, Pxx_den_ppg8 = sig.welch(ppg, fs_ppg,window='hamming',scaling='spectrum',nperseg=nn_ppg/8,nfft=nfft_pad_PPG)
f_PPG10, Pxx_den_ppg10 = sig.welch(ppg, fs_ppg,window='hamming',scaling='spectrum',nperseg=nn_ppg/10,nfft=nfft_pad_PPG)


Pxx_den_PPG_norm=Pxx_den_ppg4/np.max(np.abs(Pxx_den_ppg4))
# Conversión a dB
#pxx_ppg_db = 10 * np.log10(Pxx_den_ppg)
#pxx_ppg2_db = 10 * np.log10(Pxx_den_ppg2)
pxx_ppg4_db = 10 * np.log10(Pxx_den_PPG_norm)
#pxx_ppg8_db = 10 * np.log10(Pxx_den_ppg8)
#pxx_ppg10_db = 10 * np.log10(Pxx_den_ppg10)

#Calculo de Bw
umbral_db_ppg = -40
psd_ppg_filtrado = np.copy(Pxx_den_ppg4)
psd_ppg_filtrado[pxx_ppg4_db < umbral_db_ppg] = 0
potencia_acumulada_ppg = np.cumsum(psd_ppg_filtrado)
potencia_total_ppg = potencia_acumulada_ppg[-1]
potencia_acumulada_ppg_norm = potencia_acumulada_ppg / potencia_total_ppg

#Encontrar el índice donde se alcanza el 95% de la energía
indice_bw_ppg = np.where(potencia_acumulada_ppg_norm >= 0.95)[0][0]

#Buscar a qué frecuencia corresponde ese índice
bw_ppg = f_PPG4[indice_bw_ppg]

plt.figure(3, figsize=(12, 7))

#plt.plot(f_PPG, pxx_ppg_db, color='red', label=f'nperseg = {nn_ecg}', linewidth=1)
#plt.plot(f_PPG2, pxx_ppg2_db, color='b', label=f'nperseg = {nn_ecg//2}', linewidth=1)
plt.plot(f_PPG4, pxx_ppg4_db, color='green', label=f'nperseg = {nn_ppg//4}', linewidth=1)
#plt.plot(f_PPG8, pxx_ppg8_db, color='violet', label=f'nperseg = {nn_ecg//8}', linewidth=1)
#plt.plot(f_PPG10, pxx_ppg10_db, color='green', label=f'nperseg = {nn_ecg//10}', linewidth=1)


plt.axvline(bw_ppg, color='brown', linestyle='-', linewidth=1.5, label=f"F (95% BW): {bw_ppg: .2f} Hz")
plt.title(f"Espectro de Potencia - PPG (con Zero-Padding) Bw: {bw_ppg: .2f} Hz")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend() # Añadido para mostrar las etiquetas
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()