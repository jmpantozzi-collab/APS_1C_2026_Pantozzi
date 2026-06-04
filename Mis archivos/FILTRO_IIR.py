# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:44:23 2026

@author: jerem
"""

#%% Librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

#%% Parámetros del filtro Pasa Banda (ECG)
fs = 500  # Frecuencia de muestreo
nyq = fs / 2  # Frecuencia de Nyquist

# Especificaciones de la plantilla (en Hz)
wp1 = 1  
ws1 = 0.1
wp2 = 35 
ws2 = 45

wp = [wp1, wp2] 
ws = [ws1, ws2]
gpass = 0.5  # Atenuación máxima permitida en la banda de paso (dB)
gstop = 50   # Atenuación mínima requerida en la banda de rechazo (dB)

ftypes='butter'
#ftypes='cheby1'
#ftypes='cheby2'
#ftypes='cauer'

# Diseño del filtro usando SOS (Second-Order Sections) por estabilidad numérica
sos_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype=ftypes, output='sos')

# Calculamos el orden del filtro a partir de las secciones de segundo orden
orden_filtro = 2 * sos_coeffs.shape[0]

# Respuesta en frecuencia del filtro diseñado
w, h = sig.sosfreqz(sos_coeffs, worN=2048, fs=fs)
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
ax1.set_xlabel('Frecuencia normalizada a Nyq [#]', fontsize=10)
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