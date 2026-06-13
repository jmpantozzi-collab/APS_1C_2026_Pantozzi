# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:44:23 2026

@author: jerem
"""

#%% Librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

import scipy.io as sio
from scipy.io.wavfile import write

#%% Parámetros del filtro Pasa Banda (ECG)
fs = 1000  # Frecuencia de muestreo
nyq = fs / 2  # Frecuencia de Nyquist

# Especificaciones de la plantilla (en Hz)
wp1 = 0.5 
ws1 = 0.1
wp2 = 35 
ws2 = 45

wp = [wp1, wp2] 
ws = [ws1, ws2]
gpass = 0.5  # Atenuación máxima permitida en la banda de paso (dB)
gstop = 50   # Atenuación mínima requerida en la banda de rechazo (

ftypes='butter'
ftypesb='butter'
ftypesch1='cheby1'
ftypesch2='cheby2'
ftypesca='cauer'

# Diseño del filtro usando SOS (Second-Order Sections) por estabilidad numérica
sos_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype=ftypes, output='sos')
#sos_coeffs_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftypes, output='sos')#como uso doble filtro tengo que dividir por dos la atenuacion
sos_butter_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftypesb, output='sos')
sos_cheby1_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftypesch1, output='sos')
sos_cheby2_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftypesch2, output='sos')
sos_cauer_ff = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype=ftypesca, output='sos')
# Calculamos el orden del filtro a partir de las secciones de segundo orden
orden_filtro = 2 * sos_coeffs.shape[0]

#Parametrizar worN
ww = np.concatenate([
    np.logspace(start=-2, stop=0.1, num=500),
    np.linspace(start=1.26, stop=35, num=200),
    np.logspace(start=1.55, stop=1.65, num=300),
    np.linspace(start=46, stop=fs//2, num=50)
])

# Respuesta en frecuencia del filtro diseñado
w, h = sig.sosfreqz(sos_coeffs, worN=ww, fs=fs)
w_norm = w / nyq  # Frecuencia normalizada a Nyquist (0 a 1)




#%% ==========================================
# 1. GRÁFICO: PLANTILLA DE DISEÑO
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6), tight_layout=True)

# Magnitud en dB
def_db = 20 * np.log10(np.abs(h))
#si quisiera la frecuencia normalizada a nyquist:
#ax1.plot(w_norm, def_db, color='C0', lw=2, label=f'{ftypes}_ord_{orden_filtro}_digital')
#sino:
ax1.plot(w, def_db, color='C0', lw=2, label=f'{ftypes}_ord_{orden_filtro}_digital')

# Dibujar zonas prohibidas de la plantilla (Sombreado con patrones hachados)
# Banda de rechazo izquierda (0 a ws1) o hasta ws1/nyq si tuviera la frecuencia normalizada a nyquist
ax1.fill_between([0, ws1], -gstop, 10, color='gray', alpha=0.2, hatch='XX', label='Plantilla (Zonas prohibidas)')
# Banda de rechazo derecha (ws2 a fs/2)
ax1.fill_between([ws2, fs/2], -gstop, 10, color='gray', alpha=0.2, hatch='XX')
# Banda de paso (límite inferior de ripple entre wp1 y wp2)
ax1.fill_between([wp1, wp2], -100, -gpass, color='gray', alpha=0.15, hatch='//')

# Dibujar líneas guía para las frecuencias de corte de la plantilla
ax1.axvline(wp1, color='red', linestyle='--', alpha=0.6, lw=1)
ax1.axvline(wp2, color='red', linestyle='--', alpha=0.6, lw=1)
ax1.axhline(-gpass, color='black', linestyle=':', alpha=0.5)
ax1.axhline(-gstop, color='black', linestyle=':', alpha=0.5)

# Configuración de ejes e títulos
ax1.set_title('Plantilla de Diseño - Filtro Pasa Banda ECG', fontsize=12)
ax1.set_xlabel('Frecuencia [Hz]', fontsize=10)
ax1.set_ylabel('Amplitud [dB]', fontsize=10)
ax1.set_xlim([0, 50])  # Acercamos el zoom para ver en detalle el ECG (hasta 0.3 de Nyquist = 75Hz)
ax1.set_ylim([-100, 5])
ax1.grid(True, which='both', linestyle='-', alpha=0.5)
ax1.legend(loc='lower left')

#%% ==========================================
# 2. GRÁFICO: MAPA DE POLOS Y CEROS
# ==========================================
# Extraemos los polos (p) y ceros (z) del filtro SOS
z, p, k = sig.sos2zpk(sos_coeffs)

fig2, ax2 = plt.subplots(figsize=(6, 6), tight_layout=True)

# Dibujar el círculo unitario
theta = np.linspace(0, 2*np.pi, 400)
ax2.plot(np.cos(theta), np.sin(theta), linestyle=':', color='black', alpha=0.6, label='Círculo unitario')

# Dibujar ejes cartesianos (Real e Imaginario)
ax2.axhline(0, color='gray', lw=1, alpha=0.5)
ax2.axvline(0, color='gray', lw=1, alpha=0.5)

# Graficar Polos (X) y Ceros (O)
ax2.scatter(np.real(z), np.imag(z), s=90, marker='o', facecolors='none', edgecolors='C0', lw=2, label='Ceros ($\circ$)')
ax2.scatter(np.real(p), np.imag(p), s=90, marker='x', color='C1', lw=2, label='Polos ($\\times$)')

# Ajustes visuales de anotación 
for i, polo in enumerate(p[:2]): # Anotamos solo el primer par conjugado para no saturar la gráfica
    if np.imag(polo) > 0:
        r = np.abs(polo)
        f_polo = np.angle(polo) * fs / (2 * np.pi)
        ax2.annotate(f'f = {f_polo:.2f} Hz\n|z| = {r:.3f}', 
                     xy=(np.real(polo), np.imag(polo)), 
                     xytext=(np.real(polo)-0.4, np.imag(polo)+0.1),
                     bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
                     arrowprops=dict(arrowstyle="->", color="black"))

# Configuración del mapa de polos y ceros
ax2.set_title('Mapa de Polos y Ceros (Plano Z)', fontsize=12)
ax2.set_xlabel('$\Re(z)$ (Parte Real)')
ax2.set_ylabel('$\Im(z)$ (Parte Imaginaria)')
ax2.set_xlim([-1.1, 1.1])
ax2.set_ylim([-1.1, 1.1])
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.axis('equal') # Mantiene la relación de aspecto completamente circular
ax2.legend(loc='upper left')

plt.show()

#%% ==========================================
# 3. MUESTRA ADICIONAL: MAGNITUD Y FASE TRADICIONAL
# ==========================================
fig3, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), tight_layout=True)
ax_mag, ax_phase = axs

ax_mag.plot(w, def_db, 'C0')
ax_mag.set_title(f"Respuesta Completa del Filtro IIR Butterworth (Orden {orden_filtro})")
ax_mag.set_ylabel("Magnitud [dB]")
ax_mag.grid(True)

phase = np.unwrap(np.angle(h))
ax_phase.plot(w, phase, 'C1')
ax_phase.set_ylabel('Fase [rad]')
ax_phase.set_xlabel("Frecuencia [Hz]")
ax_phase.grid(True)

#%% ==========================================
# 4. GRÁFICO: RETARDO DE GRUPO (Formato Rad/Seg)
# ==========================================

# 1. Convertimos SOS a transferencia (b, a)
b, a = sig.sos2tf(sos_coeffs)

# 2. Calculamos el retardo de grupo (OMITIENDO 'fs' para obtener rad/seg y muestras)
# w_rad va de 0 a pi. gd_muestras es el retardo en número de muestras.
w_rad, gd_muestras = sig.group_delay((b, a), w=2048)

# 3. Convertimos el retardo de "muestras" a "segundos" dividiendo por fs
gd_seg = gd_muestras / fs

# 4. Armamos el gráfico con la estética de tu referencia
fig4, ax4 = plt.subplots(figsize=(8, 5), tight_layout=True)

# Graficamos la curva del filtro diseñado
ax4.plot(w_rad, gd_seg, color='C0', lw=2, label=f'{ftypes}_ord_{orden_filtro}_digital')

# Ajustes de títulos y etiquetas idénticos a tu imagen
ax4.set_title('Retardo de grupo', fontsize=12)
ax4.set_xlabel('Frecuencia angular [rad/seg]', fontsize=10)
ax4.set_ylabel('Retardo de grupo [seg]', fontsize=10)

# Límites del eje X estrictamente de 0 a pi (~3.14)
ax4.set_xlim([0, np.pi])

# Dejamos que el eje Y se acomode solo o podés fijar el piso en 0 si lo preferís:
# ax4.set_ylim(bottom=0)

ax4.grid(True, which='both', linestyle='-', alpha=0.7)
ax4.legend(loc='upper right')

plt.show()


#%% Procesar el ECG

fs_ecg= 1000 #Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./TS5\ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)

#Procesamiento
#verificar si el espectro nos da bien con los filtros, 
#spoiler nos da bien pero el retardo me modifica la forma por eso lo vemos feo (muy cambiada la morfologia)
#con el doble filtrdao sosfiltfilt veo la misma forma, con cuidado con la plantilla, ahora como filtro dos veces
#necesito la mitad de atenuacion gpass/2 y gstop/2
#yy= sig.sosfilt(sos_coeffs, ecg_one_lead, axis=0)
yy_butter_ff= sig.sosfiltfilt(sos_butter_ff, ecg_one_lead, axis=0)
yy_ch1_ff= sig.sosfiltfilt(sos_cheby1_ff, ecg_one_lead, axis=0)
yy_ch2_ff= sig.sosfiltfilt(sos_cheby2_ff, ecg_one_lead, axis=0)
yy_cauer_ff= sig.sosfiltfilt(sos_cauer_ff, ecg_one_lead, axis=0)

fig5, ax5 = plt.subplots(figsize=(6, 6), tight_layout=True)
plt.plot(ecg_one_lead)
plt.plot(yy_butter_ff, label='butter')
plt.plot(yy_ch1_ff, label='cheby 1')
plt.plot(yy_ch2_ff, label='cheby 2')
plt.plot(yy_cauer_ff, label='cauer')
plt.legend(loc='upper right', fontsize=10) 

plt.title('Comparación de Filtros (Filtrado de Fase Cero - sosfiltfilt)')
plt.grid(True, alpha=0.3)
plt.show()


#%% ANALISIS POR REGIONES

cant_muestras=len(ecg_one_lead)

###################################
#%% Regiones de interés sin ruido
###################################
regs_interes_sin_ruido = (
    [4000, 5500], # muestras
    [10000, 11000], # muestras (evitamos el 10e3 para que sea un entero limpio)
)

for nr, ii in enumerate(regs_interes_sin_ruido):
    # Limitamos el intervalo entre 0 y el total de la señal usando 'cant_muestras'
    inicio = int(np.max([0, ii[0]]))
    fin = int(np.min([cant_muestras, ii[1]]))
    zoom_region = np.arange(inicio, fin, dtype='uint')
   
    # Creamos una figura nueva para cada región para que no se pisen
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    
    ax.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=1.5, alpha=0.7)
    ax.plot(zoom_region, yy_butter_ff[zoom_region], label='Butterworth', linewidth=1.5)
    ax.plot(zoom_region, yy_ch2_ff[zoom_region], label='Cheby 2', linewidth=1.5)
    ax.plot(zoom_region, yy_cauer_ff[zoom_region], label='Cauer / Elíptico', linewidth=1.5)
   
    ax.set_title(f'Región sin Ruido {nr+1}: Muestras {inicio} a {fin}')
    ax.set_ylabel('Amplitud [Adimensional]')
    ax.set_xlabel('Muestras (#)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.show()

###################################
#%% Regiones de interés con ruido
###################################
regs_interes_ruido = (
    np.array([5, 5.2]) * 60 * fs_ecg,   # minutos a muestras
    np.array([12, 12.4]) * 60 * fs_ecg, # minutos a muestras
    np.array([15, 15.2]) * 60 * fs_ecg, # minutos a muestras
)

for nr, ii in enumerate(regs_interes_ruido):
    # Limitamos el intervalo entre 0 y el total de la señal usando 'cant_muestras'
    inicio = int(np.max([0, ii[0]]))
    fin = int(np.min([cant_muestras, ii[1]]))
    zoom_region = np.arange(inicio, fin, dtype='uint')
   
    # Creamos una figura nueva para cada región
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    
    ax.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=1.5, alpha=0.7)
    ax.plot(zoom_region, yy_butter_ff[zoom_region], label='Butter', linewidth=1.5)
    ax.plot(zoom_region, yy_ch2_ff[zoom_region], label='Cheby 2', linewidth=1.5)
    ax.plot(zoom_region, yy_cauer_ff[zoom_region], label='Cauer', linewidth=1.5)
   
    ax.set_title(f'Región con ruido {nr+1}: Muestras {inicio} a {fin}')
    ax.set_ylabel('Amplitud [Adimensional]')
    ax.set_xlabel('Muestras (#)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
           
    plt.show()