# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:16:08 2026

@author: jerem
"""
#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

eps = 1e-12
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx

#%% GENERO MI SEÑAL x(n)
n=1000
a0=np.sqrt(2)
fs=n
delf=fs/n
rea=200

SNR=10
Pr=10**(-SNR/10)
R= np.random.normal(0, np.sqrt(Pr), n)

fr = np.random.uniform(-2, 2, rea)

#Con cuidado pasar omega a frec.... pi/2=n/4 2pi/n=1 y deltaf=fs/n
#w1= n/4 + fr*((2*np.pi)/n)
w1= (n/4 + fr)*delf

arr_w1=w1.reshape(rea,1)
nn=np.arange(0,n)
arr_n=nn.reshape(1,n)

# matriz_w1=np.tile(arr_w1,(1, n))
# matriz_n=np.tile(arr_n, (rea, 1))

# f=(matriz_w1)*(matriz_n)

f=arr_w1*arr_n*1/fs
s=a0*np.sin(2*np.pi*f)


x=s+R


#%% FFTS

#ventanas
flat=sig.windows.flattop(n)

#Amplitud a1 ventana rectangular
half_n = n // 2

x_ven=x
a1=np.fft.fft(x_ven,axis=1)* (1/n) #escalamos con 1/n para ver W

a1_half = a1[:,:half_n]
# Multiplicamos por 2 para recuperar la amplitud de la sinusoide
# (porque la FFT reparte la energía en frecuencias + y -)
a1mod= np.abs(a1_half)**2
a1pot=a1mod*2

a1mod_dB = 10 * np.log10(a1pot+eps)





#%%Estimadores
valor_teorico = np.sqrt(2) / 2  # a0 / 2
#Voy a agarrar la feta de omega 0 para ver la distribucion de los valores que me dieron todas
#las realizaciones para ver como es el estimador a1

#Histograma rectangular

# Extraer la columna 250 de la matriz de amplitud (módulo)
# Tomamos todas las realizaciones (filas) de la columna 250
estimador_a1 = np.abs(a1[:, 250])
media = np.mean(estimador_a1)
desvio = np.std(estimador_a1)

print(f"Estimador final: {media:.4f} ± {desvio:.4f}")



#%% Graficar

# Elegimos qué fila queremos ver (por ejemplo, la primera)
indice_a_graficar = 0 

# Extraemos la fila completa
# El ':' significa "todas las columnas"
seno_seleccionado = x[indice_a_graficar, :]

# Graficamos

plt.figure(1, figsize=(10, 6))
plt.plot(seno_seleccionado)
plt.title(f"Seno número {indice_a_graficar} de la matriz")
plt.xlabel("Muestras (n)")
plt.ylabel("Amplitud")
plt.grid(True)

plt.show()

plt.figure(2, figsize=(10, 6))
plt.clf()
plt.plot(a1mod_dB.T)
plt.grid(True)
plt.show()

#histograma grafico ventana
plt.figure(3, figsize=(10, 6))
# Histograma
n_bins, bins, patches = plt.hist(estimador_a1, bins=30, color='orange', 
                                 edgecolor='white', alpha=0.6, label='Realizaciones')

# Línea de la Media
plt.axvline(media, color='blue', linestyle='-', linewidth=2.5, 
            label=f'Media ($\mu$): {media:.3f}')

# Línea del Valor Teórico
plt.axvline(valor_teorico, color='red', linestyle='--', linewidth=2, 
            label=f'Teórico: {valor_teorico:.3f}')

# Zona de Desvío Estándar (Media ± 1 Desvío)
plt.axvspan(media - desvio, media + desvio, color='blue', 
            alpha=0.15, label=f'Desvío ($\sigma$): ±{desvio:.3f}')

# 3. Estética y etiquetas
plt.title(r'Análisis Estadístico del Estimador $\hat{a}_1$', fontsize=14)
plt.xlabel('Amplitud Lineal', fontsize=12)
plt.ylabel('Frecuencia (Realizaciones)', fontsize=12)

# Añadir un cuadro de texto con los resultados numéricos exactos
texto_stats = f'Estadísticas:\n$\mu$ = {media:.4f}\n$\sigma$ = {desvio:.4f}\nBias = {media - valor_teorico:.4f}'
plt.annotate(texto_stats, xy=(0.05, 0.7), xycoords='axes fraction', 
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()




