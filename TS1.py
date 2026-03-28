# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:33:02 2026

Ejercicio 1: 
    
Sintetizar:
-Señal sinusoidal de 2KHz.
-Misma señal 3 dB y desfasada en π/2.
-Misma señal modulada en amplitud por otra señal sinusoidal de frecuencia de 1000 KHz.
-Misma señal con efecto de saturación al 75% de su amplitud. Ayuda: ver numpy.clip().
-Una señal cuadrada de 4KHz.
-Un pulso rectangular de 10ms.
En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda.

Ejercicio 2: 
    
Dado h[n] = δ[n] - δ[n - 4], encontrar y[n] = x[n] * h[n] para cada una de las siguientes x[n]:

a) x[n] = cos(ω₀.n. TS). Expresar la respuesta como un único coseno de la forma A cos(ω₀. n . TS + φ).

b) x[n] = (1/2)ⁿ u[n].

c) x[n] = u[n + 1] - u[n - 2].
    
@author: jeremias Pantozzi
"""
#%% Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal as sig
#%% Funciones 

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
        tt= np.arange(nn)/fs
        xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
        return tt, xx
 
def mi_funcion_cua (vmax,ff, nn, fs):
    n = np.arange(nn)
    tt= n/fs
    y= vmax * np.sign(np.sin(2 * np.pi * ff * n / fs))
    return tt, y
def pulso (vmax, T, nn, fs): #vmax: amplitud, T:duracion dle pulso [seg], nn: cantd. muestras, fs: frecuencia de muestreo
    tt = np.arange(nn) / fs
    x = np.where(tt < T, vmax, 0)
    return tt, x

    
#%% Aplicacion

#%%Señal Sen de 2kHz

ff=2000
muestras_xciclo=40 #cantidad de muestras que quiero por ciclo, respetando niquitz minimo necesito 2.
fs=muestras_xciclo*ff #frecuencia de muestreo para respetar las muestras por ciclo
nn= 4*muestras_xciclo #cuantos ciclos quiero visualizar, en este caso 4.

tt,xx=mi_funcion_sen(np.sqrt(2),0,ff,0,nn,fs)



#%%Señal amplificada y desfasada
ph=np.pi/2 #desfase que nos piden
k=10**(3/20) #amplificacion 3dB = 20log(A/A0) 
tt1,xx1=mi_funcion_sen(k*np.sqrt(2),0,ff,ph,nn,fs)

plt.figure(0)
plt.clf()
plt.plot(tt*1000, xx , label='2 kHz original')
plt.plot(tt1*1000, xx1, label='Señal +3 dB y desfasada π/2')
plt.legend()
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# --- Datos ---
periodo = 1 / ff
tipo_senal = "Señal de potencia"
# Potencias
P1 = (np.sqrt(2)**2) / 2
P2 = (k*np.sqrt(2))**2 / 2

texto_original = (
    f"Original\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Potencia = {P1:.3f} W\n"
    f"Señal de potencia"
)

texto_amp = (
    f"+3 dB y desfase π/2\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Potencia = {P2:.3f} W\n"
    f"Señal de potencia"
)

# Anotación cerca de cada curva
plt.annotate(
    texto_original,
    xy=(tt[int(nn*0.1)], xx[int(nn*0.1)]),
    xytext=(20, 20),
    textcoords='offset points',
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.annotate(
    texto_amp,
    xy=(tt1[int(nn*0.6)], xx1[int(nn*0.6)]),
    xytext=(20, -40),
    textcoords='offset points',
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.7)
)


plt.show()


#%%Modulacion de sen original con otra de frecuancia 1000Hz (Asumo error de tipeo en consigna 10000Hz no es logico para modular usar una señal de mayor frecuencia que la portadora)
ff2=1000
tt2,xx2=mi_funcion_sen(1,1,ff2,0,nn,fs)

ym= xx*xx2


plt.legend()
plt.figure(1)
plt.clf()
plt.plot(tt*1000,ym, label ='Modulada')
plt.plot(tt*1000, xx , label = '2KHZ original',linestyle='dotted')
plt.plot(tt2*1000, xx2, label = 'Señal Modulante',color='green',linestyle='--')
plt.plot(tt2*1000, -xx2, label = 'Señal Modulante',color='green',linestyle='--')
plt.legend()
plt.title("Funcion en tiempo")
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# --- Datos ---
periodo = 1 / ff
tipo_senal = "Señal de potencia"
# Potencias
P1 = (np.sqrt(2)**2) / 2
P2 = (k*np.sqrt(2))**2 / 2

texto_original = (
    f"Original\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Señal de potencia"
)

texto_amp = (
    f"Modulante\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Señal de potencia"
)
texto_mod = (
    f"Modulada\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Señal de potencia"
)


# Anotación cerca de cada curva
plt.annotate(
    texto_original,
    xy=(tt[int(nn*0.1)], xx[int(nn*0.1)]),
    xytext=(20, 20),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.annotate(
    texto_amp,
    xy=(tt1[int(nn*0.6)], xx1[int(nn*0.6)]),
    xytext=(20, -40),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.annotate(
    texto_mod,
    xy=(tt1[int(nn*0.6)], xx1[int(nn*0.6)]),
    xytext=(120, -40),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.show()

# FFT
XX = np.fft.fft(xx)/nn
YY = np.fft.fft(ym)/nn

# Frecuencias
freqs = np.fft.fftfreq(len(xx), 1/fs)

# Centrado
XX_shift = np.fft.fftshift(XX)
YY_shift = np.fft.fftshift(YY)
freqs_shift = np.fft.fftshift(freqs)

# Magnitud en dB
XX_dB = 20*np.log10(np.abs(XX_shift))
YY_dB = 20*np.log10(np.abs(YY_shift))

plt.figure()
plt.plot(freqs_shift, XX_dB, label="Original")
plt.plot(freqs_shift, YY_dB, label="Modulada",linestyle='dotted')
plt.grid(True)
plt.legend()
plt.title("Espectro en frecuencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.show()



#%%Señal anterior recortada al 75% de su amplitud. En este caso 75% de raiz de 2

lim=0.75*np.sqrt(3)

y_rec=np.clip(xx, -lim, lim) #uso la funcion clipeo, si la señal xx supera algun limite se reemplaza por el limite

plt.figure(3)
plt.clf()
plt.plot(tt*1000, xx , label = '2KHZ original',color='blue',linestyle='dotted')
plt.plot(tt*1000,y_rec, label ='Recortada', color='red')
plt.legend()
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

texto_recot = (
    f"Recortada\n"
    f"T = {periodo*1000:.3f} ms\n"
    f"Muestras = {nn}\n"
    f"Señal de potencia"
)


# Anotación cerca de cada curva
plt.annotate(
    texto_recot,
    xy=(tt[int(nn*0.1)], xx[int(nn*0.1)]),
    xytext=(20, 20),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.show()

#%%Onda cuadrada de 4Khz

ffc=4000
muestras_xciclo=40 #cantidad de muestras que quiero por ciclo, respetando niquitz minimo necesito 2.
fsc=muestras_xciclo*ffc #frecuencia de muestreo para respetar las muestras por ciclo
nnc= 4*muestras_xciclo #cuantos ciclos quiero visualizar, en este caso 4.

tc, yc= mi_funcion_cua(np.sqrt(2), ffc, nnc, fsc)

plt.figure(4)
plt.clf()
plt.plot(tc*1000,yc, label ='Onda cuadrda')
plt.legend()
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# --- Datos ---
periodoc = 1 / ffc
texto_cua = (
    f"Cuadrada\n"
    f"T = {periodoc*1000:.3f} ms\n"
    f"Muestras = {nnc}\n"
    f"Señal de potencia"
)


# Anotación cerca de cada curva
plt.annotate(
    texto_cua,
    xy=(tt[int(nn*0.1)], xx[int(nn*0.1)]),
    xytext=(20, 20),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.show()


#%% Pulso de 10ms

vmax=1.5
T=10/1000 #10ms a segundos

ttp,yp= pulso(vmax,T,500,2000)

plt.figure(5)
plt.clf()
plt.plot(ttp,yp, label ='pulso de 10ms')
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# --- Datos ---

texto_puls = (
    f"Pulso\n"
    f"T = {T*1000:.1f} ms\n"
    f"Muestras = {500}\n"
    f"Señal de Energia"
)


# Anotación cerca de cada curva
plt.annotate(
    texto_puls,
    xy=(tt[int(nn*0.1)], xx[int(nn*0.1)]),
    xytext=(20, 20),
    textcoords='offset points',
    fontsize=7,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.show()

#%%Verificacion de ejercicio 2
#%% como usar convolucionar, primero creo una delta
T0=1/fs
w0=2 * np.pi * ff
n = np.arange(nn)
#creamos h[x]=delta(n)-delta(n-4)
h= np.zeros(nn)
h[0]= 1
h[4]=-1

# --- (a) ---

tt= np.arange(nn)/fs
xx = np.cos(w0*n*T0)

#mi solucion
x1=2*np.sin(2*w0*T0)*np.cos(w0*n*T0-2*w0*T0+(np.pi/2))

yy= sig.convolve(xx,h)

plt.figure(6)
plt.clf()
plt.plot(x1[:175],label ='mi solucion',color='red', linestyle='dotted')
plt.plot(yy[:175],label ='convolucion por codigo',color='pink',linestyle='dotted')
plt.plot(h[:175],label ='h[n]',color='violet', linestyle='--')
plt.plot(xx[:175],label ='x[n]',color='green', linestyle='dotted')
plt.title('convolucion ejercicio a')
plt.legend()
plt.show()

u = lambda x: (x >= 0).astype(int)

# --- (b) ---
xxb = (0.5)**n * u(n)

x2 = (0.5)**n * u(n) - (0.5)**(n-4) * u(n-4)

yy2 = sig.convolve(xxb, h)

plt.figure(7)
plt.clf()
plt.plot(x2[:25], label='mi solución', color='red', linestyle='dotted')
plt.plot(yy2[:25], label='convolución por código', color='pink', linestyle='dotted')
plt.plot(xxb[:25], label='x[n]', color='green', linestyle='dotted')
plt.plot(h[:25],label ='h[n]',color='violet', linestyle='--')
plt.title('convolución ejercicio b')
plt.legend()
plt.grid(True)
plt.show()

# --- (c) ---
xxc = u(n+1) - u(n-2)

x3 = (u(n+1) - u(n-2)) - (u(n-3) - u(n-6))

yy3 = sig.convolve(xxc, h)

plt.figure(8)
plt.clf()
plt.plot(x3[:25], label='mi solución', color='red', linestyle='dotted')
plt.plot(yy3[:25], label='convolución por código', color='pink', linestyle='dotted')
plt.plot(xxc[:25], label='x[n]', color='green', linestyle='dotted')
plt.plot(h[:25],label ='h[n]',color='violet', linestyle='--')
plt.title('convolución ejercicio c')
plt.legend()
plt.grid(True)
plt.show()

