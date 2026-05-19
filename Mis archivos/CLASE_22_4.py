import matplotlib.pyplot as plt
import numpy as np
import math 
from math import log 

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx

vmax = np.sqrt(2) # para que la potencia sea 1 W
dc=0
ph= 0 
N= 1000
fs= N
df= fs/N
eps = 1e-12
frec = np.arange(0, fs/2, df)

mu = 0
ff1=(N/4)*df
tt,xx1 = mi_funcion_sen(vmax, dc,ff1,ph,N,fs)

ff2=((N/4) + 0.25)*df
tt,xx2 = mi_funcion_sen(vmax, dc,ff2,ph,N,fs)

ff3=((N/4) + 0.5)*df
tt,xx3= mi_funcion_sen(vmax, dc,ff3,ph,N,fs)


XX1 = np.fft.fft(xx1)
XXmod = np.abs(XX1) / N # MODULO
XXmod_cuadrado = XXmod**2 # MODULO AL CUADRADO -> ESPECTRO POTENCIA
XXesp_db = 10 * np.log10(XXmod_cuadrado + 1e-12 ) # ESPECTRO POTENCIA 

XX2 = np.fft.fft(xx2)
XXmod2 = np.abs(XX2) / N # MODULO
XXmod_cuadrado2 = XXmod2**2 # MODULO AL CUADRADO -> ESPECTRO POTENCIA
XXesp_db2 = 10 * np.log10(XXmod_cuadrado2 + 1e-12 ) # ESPECTRO POTENCIA 

XX3 = np.fft.fft(xx3)
XXmod3 = np.abs(XX3) / N # MODULO
XXmod_cuadrado3 = XXmod3**2 # MODULO AL CUADRADO -> ESPECTRO POTENCIA
XXesp_db3 = 10 * np.log10(XXmod_cuadrado3 + 1e-12 ) # ESPECTRO POTENCIA 

# YY_mitad = YYesp_db[0:500] # PARA GRAFICAR LA MITAD 

# --- 3. ZERO PADDING MANUAL (CONCATENACIÓN) ---
# Creamos un vector de ceros de tamaño 9*N
vector_ceros = np.zeros(9 * N)

# Concatenamos la señal original (xx3) con los ceros
# Esto equivale a decir: señal_larga = [señal_original, 0, 0, 0, ...]
xx3_padded = np.concatenate((xx3, vector_ceros))

# Actualizamos parámetros para la nueva señal
N_total = len(xx3_padded)
frec_pad = np.arange(0, fs/2, fs/N_total)

# --- 4. CÁLCULO DE ESPECTROS (Normalizados a 0 dB) ---

# Caso N/4 (para referencia)
XX1 = np.fft.fft(xx1)
mag1 = (np.abs(XX1) / N) * 2 # Factor 2 para compensar freq. negativas
db1 = 10 * np.log10(mag1[:N//2]**2 + eps)

# Caso N/4 + 0.5 (Original)
XX3 = np.fft.fft(xx3)
mag3 = (np.abs(XX3) / N) * 2
db3 = 10 * np.log10(mag3[:N//2]**2 + eps)

# Caso N/4 + 0.5 (Con Padding Manual)
XX_pad = np.fft.fft(xx3_padded)
# IMPORTANTE: Dividimos por N (el original) para que la amplitud sea la misma
mag_pad = (np.abs(XX_pad) / N) * 2
db_pad = 10 * np.log10(mag_pad[:N_total//2]**2 + eps)

# --- 5. GRÁFICOS ---
plt.figure(figsize=(12, 10))

# Subplot 1: Los tres puntos del inciso A
plt.subplot(2, 1, 1)
plt.plot(frec, db1, label='Senoidal N/4 (0 dB Ref)', color='navy')
plt.plot(frec, db3, label='Senoidal N/4 + 0.5', color='crimson')
plt.scatter(frec, db3, color='crimson', s=15)
plt.title('Punto A: Espectro con normalización corregida (0 dB)')
plt.ylabel('Magnitud [dB]')
plt.xlim(240, 260)
plt.ylim(-60, 5)
plt.grid(True)
plt.legend()

# Subplot 2: El efecto del Padding manual
plt.subplot(2, 1, 2)
plt.plot(frec_pad, db_pad, color='green', label='Padding por Concatenación (10*N)')
plt.scatter(frec, db3, color='crimson', s=25, label='Bins originales', zorder=3)
plt.title('Punto C: Revelando los lóbulos mediante concatenación de ceros')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(240, 260)
plt.ylim(-60, 5)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


        

