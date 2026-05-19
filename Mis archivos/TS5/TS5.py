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

#Audio

f_cuca, Pxx_den_cuca = sig.welch(wav_data_cuca, fs_audio_cuca,window='hamming',scaling='spectrum',nperseg=400)
pxx_cuca_db=10 * np.log10(Pxx_den_cuca)

plt.figure(1, figsize=(12, 7))
plt.semilogy(f_cuca, Pxx_den_cuca)
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


#ecg
f_ecg, Pxx_den_ecg = sig.welch(ecg_one_lead, fs_ecg,window='hamming',scaling='spectrum',nperseg=2500)
pxx_ecg_db=10 * np.log10(Pxx_den_ecg)


plt.figure(2, figsize=(12, 7))
plt.semilogy(f_ecg, Pxx_den_ecg)
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()