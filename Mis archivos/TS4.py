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


n=1000

eps = 1e-12
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx
def calcular_potencia_db(señal):
    XX = np.fft.fft(señal)
    XXmod = np.abs(XX) / n #Normalizamos 
    # Multiplicamos por 2 para sumar la energía de la frecuencia negativa
    potencia_lineal = (XXmod**2) * 2 
    return 10 * np.log10(potencia_lineal + eps)

#%% GENERO MI SEÑAL x(n)

a0=2
P_sin = (a0**2) / 2
fs=n
delf=fs/n
rea=200

# Calibración del ruido Gaussiano basado en la potencia real de la señal
# SNR = 10 * log10(P_sin / P_ruido) -> P_ruido = P_sin / 10^(SNR/10)
SNR=3
Pr = P_sin / (10**(SNR / 10))
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



#Amplitud a1 ventana rectangular
half_n = n // 2

x_ven=x
a1=np.fft.fft(x_ven,axis=1)* (1.0 / (n * np.sqrt(2))) #escalamos con 1/n para ver W, sqrt(2) es la correccion para que sea 1W ya que a0=2


a1_half = a1[:,:half_n]
# Multiplicamos por 2 para recuperar la amplitud de la sinusoide
# (porque la FFT reparte la energía en frecuencias + y -)
a1mod= np.abs(a1_half)**2
a1pot=a1mod*2

a1mod_dB = 10 * np.log10(a1pot+eps)






#%% ESTIMADORES
# Valor real de la amplitud de la senoidal calibrada
valor_teorico = 1.0
f0 = 250 

# Diccionarios para ir guardando los resultados de cada ventana
datos_amplitud = {}
datos_frecuencia = {}

# Listas para guardar las realizaciones crudas (útil para tus histogramas)
estimadores_ventanas = []
estimadores_frec = []

#PROCESAMIENTO DE LA VENTANA RECTANGULAR
indices_max_rect = np.argmax(np.abs(a1[:, :half_n]), axis=1)
estimador_a1 = np.abs(a1[:, 250]) * 2
estimador_f1 = indices_max_rect 

# Guardamos en las listas generales
estimadores_ventanas.append(estimador_a1)
estimadores_frec.append(estimador_f1)

# Estadísticas Rectangular
datos_amplitud["Rectangular"] = {
    "sa": np.mean(estimador_a1) - valor_teorico,
    "va": np.var(estimador_a1)
}
datos_frecuencia["Rectangular"] = {
    "sa": np.mean(estimador_f1) - f0,
    "va": np.var(estimador_f1)
}


#PROCESAMIENTO DE LAS OTRAS VENTANAS
w_flat = sig.windows.flattop(n)
w_blackmanharris = sig.windows.blackmanharris(n)
w_hamming = sig.windows.hamming(n)

ventanas = [w_flat, w_blackmanharris, w_hamming]
nombres = ["Flat-top", "Blackman-Harris", "Hamming"]

for v_actual, nombre in zip(ventanas, nombres):
    # Aplicar ventana y FFT calibrada
    x_ven = x * v_actual 
    a_fft = np.fft.fft(x_ven, axis=1) * (1.0 / (n * np.sqrt(2)))

    # Estimadores (*2 por mitad de FFT)
    est_a = np.abs(a_fft[:, 250]) * 2
    indices_max_v = np.argmax(np.abs(a_fft[:, :half_n]), axis=1)
    est_f = indices_max_v 

    # Guardar vectores crudos para los gráficos
    estimadores_ventanas.append(est_a)
    estimadores_frec.append(est_f)

    # Estadísticas Ventana Actual
    datos_amplitud[nombre] = {
        "sa": np.mean(est_a) - valor_teorico,
        "va": np.var(est_a)
    }
    datos_frecuencia[nombre] = {
        "sa": np.mean(est_f) - f0,
        "va": np.var(est_f)
    }


#IMPRESIÓN DE LAS TABLAS
print(f"\n======================================================================")
print(f"                 RESULTADOS DE SIMULACIÓN PARA SNR = {SNR} dB")
print(f"======================================================================")

print("\n--- CUADRO COMPARATIVO: ESTIMACIÓN DE AMPLITUD ---")
print(f"{'Ventana Espectral':<20} | {'Valor Medio':<14} | {'Sesgo (sa)':<14} | {'Varianza (va)':<14}")
print("-" * 72)
for ventana, metrices in datos_amplitud.items():
    # Recuperamos la media exacta calculada previamente en el bucle
    media_a = metrices['sa'] + valor_teorico
    print(f"{ventana:<20} | {media_a:<14.6f} | {metrices['sa']:<14.6f} | {metrices['va']:<14.6f}")

print("\n--- CUADRO COMPARATIVO: ESTIMACIÓN DE FRECUENCIA ---")
print(f"{'Ventana Espectral':<20} | {'Valor Medio [Hz]':<16} | {'Sesgo (sf)':<14} | {'Varianza (vf)':<14}")
print("-" * 72)
for ventana, metrices in datos_frecuencia.items():
    # Recuperamos la media en Hz calculada previamente en el bucle
    media_f_val = metrices['sa'] + f0
    print(f"{ventana:<20} | {media_f_val:<16.4f} | {metrices['sa']:<14.4f} | {metrices['va']:<14.6f}")

#VARIABLES FINALES PARA LOS GRÁFICOS
todos_los_estimadores = estimadores_ventanas
todos_est_frec = estimadores_frec
nombres_plot = ["Rectangular", "Flat-top", "Blackman-Harris", "Hamming"]
#%% Graficar

# --- Figura 1: Señal en el tiempo ---
indice_a_graficar = 0 
seno_seleccionado = x[indice_a_graficar, :]

plt.figure(1, figsize=(10, 4))
plt.plot(nn, seno_seleccionado)
plt.title(f"Señal Temporal (Realización {indice_a_graficar}) - SNR: {SNR} dB")
plt.xlabel("Muestras (n)")
plt.ylabel("Amplitud")
plt.grid(True, alpha=0.3)
plt.show()

# --- Figura 2: Espectro en dB ---
plt.figure(2, figsize=(10, 4))
# Graficamos el espectro de todas las realizaciones para ver cómo "baila" el pico
plt.plot(a1mod_dB.T, alpha=0.1, color='gray') 
plt.plot(a1mod_dB[0, :], color='blue', label='Una realización') # Resaltamos una
plt.title("Espectro de Amplitud (dB)")
plt.xlabel("Bins de Frecuencia")
plt.ylabel("Potencia [dB]")
plt.xlim(200, 300) # Hacemos zoom alrededor del bin 250
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#%% Figura 3: Grafico Comparativo de Histogramas Amplitud

plt.figure(3, figsize=(12, 7))
colores = ['orange', 'blue', 'green', 'purple']

# Altura para las líneas de rango (para que no se encimen en el piso)
alturas_lineas = [1, 2, 3, 4] 

for est, nombre, color, h in zip(todos_los_estimadores, nombres_plot, colores, alturas_lineas):
    # 1. Graficar histograma
    plt.hist(est, bins=30, alpha=0.3, label=nombre, color=color, edgecolor='black')
    
    # 2. Calcular límites del histograma actual
    min_val = np.min(est)
    max_val = np.max(est)
    media_val = np.mean(est)

    # 3. Dibujar línea de la Media
    plt.axvline(media_val, color=color, linestyle='-', linewidth=2)

    # 4. Dibujar línea de Rango (donde arranca y termina el histograma)
    # Ponemos la línea un poquito arriba del cero (h) para que se distingan entre sí
    plt.hlines(y=h, xmin=min_val, xmax=max_val, color=color, linewidth=4, alpha=0.8)
    
    # Marcadores en las puntas (opcional, para que quede más claro)
    plt.plot(min_val, h, '|', color=color, markersize=10, markeredgewidth=3)
    plt.plot(max_val, h, '|', color=color, markersize=10, markeredgewidth=3)

# Línea del Valor Teórico
plt.axvline(valor_teorico, color='red', linestyle='--', linewidth=3, 
            label=f'Teórico ($a_0$): {valor_teorico:.3f}')

plt.title('Dispersión de los Estimadores (Rango y Media)', fontsize=14)
plt.xlabel('Amplitud Lineal (Volts)', fontsize=12)
plt.ylabel('Cantidad de Realizaciones', fontsize=12)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.2)

# Ajuste automático del eje X
plt.xlim(np.min(todos_los_estimadores)*0.9, np.max(todos_los_estimadores)*1.1)

plt.tight_layout()
plt.show()

#%% Figura 4: Grafico Comparativo de Histogramas frecuencia

plt.figure(4, figsize=(12, 7))
colores = ['red', 'blue', 'green', 'purple']

# Altura para las líneas de rango (para que no se encimen en el piso)
alturas_lineas = [1, 2, 3, 4] 

for est, nombre, color, h in zip(todos_est_frec, nombres_plot, colores, alturas_lineas):
    # 1. Graficar histograma
    plt.hist(est, bins=30, alpha=0.3, label=nombre, color=color, edgecolor='black')
    

plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.2)

# Ajuste automático del eje X
plt.xlim(np.min(todos_est_frec)*0.9, np.max(todos_est_frec)*1.1)

plt.tight_layout()
plt.show()

