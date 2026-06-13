# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:13:49 2026

@author: jerem
"""

#%%  Diseño de FIRs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig 
import scipy.io as sio  

fs_ecg = 1000
mat_struct = sio.loadmat('./TS5/ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()

# --- PLANTILLA OBJETIVO (CLÍNICA) ---
wp1_d, ws1_d = 0.5, 0.1
wp2_d, ws2_d = 35.0, 45.0
gpass, gstop = 1.0, 50.0
fs = 1000

# --- PLANTILLA DISEÑADA PARA FIR MÍNIMOS CUADRADOS ---
ws1 = 0.2   # STOP baja
wp1 = 0.7  # PASO baja
wp2 = 35.0  # PASO alta
ws2 = 36  # STOP alta

# Forzamos que queden en orden estrictamente creciente para evitar ValueErrors
bandas = np.array([0.0, ws1, wp1, wp2, ws2, fs//2])
bandas = np.sort(bandas)

# Definición de ganancias por banda (0: rechazo, 1: paso)
gains = np.array([0, 0, 1, 1, 0, 0])

# Número de coeficientes (Taps) - Orden impar (Fase Lineal Tipo 1)
numtaps = 1851
demora = (numtaps - 1) // 2

# Vector de pesos (Mucha prioridad a la banda de detención baja)
pesos = np.array([2500, 1, 1])

# --- DISEÑO DEL FILTRO ---
b_cua = sig.firls(numtaps, bands=bandas, desired=gains, weight=pesos, fs=fs)

# --- RESPUESTA EN FRECUENCIA (Vector Lineal Limpio) ---
ww = np.linspace(0, fs//2, 2000)
w, h = sig.freqz(b_cua, a=1, worN=ww, fs=fs)
def_db = 20 * np.log10(np.abs(h) + 1e-12)


# ==========================================
# 1. GRÁFICO: PLANTILLA DE DISEÑO
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6), tight_layout=True)

# Magnitud en dB
def_db = 20 * np.log10(np.abs(h))
#si quisiera la frecuencia normalizada a nyquist:
#ax1.plot(w_norm, def_db, color='C0', lw=2, label=f'{ftypes}_ord_{orden_filtro}_digital')
#sino:
ax1.plot(w, def_db, color='C0', lw=2, label='FIR VENTANA')

# Dibujar zonas prohibidas de la plantilla (Sombreado con patrones hachados)
# Banda de rechazo izquierda (0 a ws1) o hasta ws1/nyq si tuviera la frecuencia normalizada a nyquist
ax1.fill_between([0, ws1_d], -gstop, 10, color='gray', alpha=0.2, hatch='XX', label='Plantilla (Zonas prohibidas)')
# Banda de rechazo derecha (ws2 a fs/2)
ax1.fill_between([ws2_d, fs/2], -gstop, 10, color='gray', alpha=0.2, hatch='XX')
# Banda de paso (límite inferior de ripple entre wp1 y wp2)
ax1.fill_between([wp1_d, wp2_d], -100, -gpass, color='gray', alpha=0.15, hatch='//')

# Dibujar líneas guía para las frecuencias de corte de la plantilla
ax1.axvline(wp1, color='red', linestyle='--', alpha=0.6, lw=1)
ax1.axvline(wp2, color='red', linestyle='--', alpha=0.6, lw=1)
ax1.axhline(-gpass, color='black', linestyle=':', alpha=0.5)
ax1.axhline(-gstop, color='black', linestyle=':', alpha=0.5)

# Configuración de ejes e títulos
ax1.set_title('Plantilla de Diseño - Filtro Pasa Banda ECG', fontsize=12)
ax1.set_xlabel('Frecuencia normalizada a Nyq [#]', fontsize=10)
ax1.set_ylabel('Amplitud [dB]', fontsize=10)
ax1.set_xlim([-10, 100])  # Acercamos el zoom para ver en detalle el ECG (hasta 0.3 de Nyquist = 75Hz)
ax1.set_ylim([-100, 5])
ax1.grid(True, which='both', linestyle='-', alpha=0.5)
ax1.legend(loc='lower left')

fase = np.unwrap(np.angle(h))

#%%
'VERIFICACION FILTRO Y ORIGINAL'

fs = 1000
ripple = 1 

ws1 = .1 #STOP
wp1 = .5 #PASO
 
ws2 = 45 #STOP
wp2 = 35 # con 99% de energia me dio 31.41 

gpass = 1
gstop = 40


wp = [wp1, wp2]
ws = [ws1, ws2]



sos_coeff_cauer = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='cauer', output='sos', fs=fs)

sos_coeff_cheby2 = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='cheby2', output='sos', fs=fs)

#yy = sig.sosfilt(sos_coeff, ecg_one_lead)

yy_cauer = sig.sosfiltfilt(sos_coeff_cauer, ecg_one_lead)

yy_cheby2 = sig.sosfiltfilt(sos_coeff_cheby2, ecg_one_lead)

cant_muestras = len(ecg_one_lead)
ecg_one_lead = ecg_one_lead


ECG_f_cauer = yy_cauer

ECG_f_cheby2 = yy_cheby2

ECG_f_cua = sig.lfilter(b_cua, 1., ecg_one_lead)

###################################
# Regiones de interés sin ruido #
###################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG SIN RUIDO', linewidth=1)
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer',  linewidth=1)
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2', linewidth=1)
    plt.plot(zoom_region, ECG_f_cua[zoom_region + demora], label='FIR Minimos Cuadrados')
   
    plt.title('ECG SIN RUIDO filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
# Regiones de interés con ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG CON RUIDO', linewidth=1)    
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer', linewidth=1)
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2', linewidth=1)
    plt.plot(zoom_region, ECG_f_cua[zoom_region + demora], label='FIR Minimos Cuadrados')
   
    plt.title('ECG CON RUIDO filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()