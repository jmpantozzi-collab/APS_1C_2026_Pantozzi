# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:31:03 2026

@author: jerem
"""
#%% Librerias
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

#%% Definiciones
#Los coeficientes del sistema
b = [1,0,0,0, -1]  # Coeficientes que multiplican a las x(n)
a = [1]           # Coeficiente que multiplica a y(n)

# Calculamos la respuesta en frecuencia
w, h = signal.freqz(b, a, worN=2000)

# Calculamos las raíces para el mapa de polos y ceros
ceros = np.roots(b)
polos = np.roots(a)

if len(a) == 1 and a[0] == 1:
    polos = np.zeros(len(b) - 1)  # Crea un array de ceros reales [0, 0, 0]
else:
    polos = np.roots(a)

#%% Graficamos Módulo y Fase
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Gráfico de Magnitud
ax1.plot(w / np.pi, np.abs(h), 'b', linewidth=2)
ax1.set_title('Respuesta en Frecuencia - Sistema C')
ax1.set_ylabel('Magnitud |T(e^{j\omega})|')
ax1.grid(True)

# Gráfico de Fase
ax2.plot(w / np.pi, np.angle(h), 'r', linewidth=2)
ax2.set_xlabel('Frecuencia Normalizada (\times \pi rad/muestra)')
ax2.set_ylabel('Fase (radianes)')
ax2.grid(True)
plt.tight_layout()

# Mapa de Polos y Ceros (Plano Z)
fig2, ax3 = plt.subplots(figsize=(6, 6))

# Dibujamos el círculo unitario punteado al fondo (zorder=1)
theta = np.linspace(0, 2 * np.pi, 200)
ax3.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Círculo unitario', zorder=1)

# Dibujamos los ejes cartesianos al fondo (zorder=1)
ax3.axhline(0, color='gray', linewidth=0.8, zorder=1)
ax3.axvline(0, color='gray', linewidth=0.8, zorder=1)

# Graficamos los Ceros adelante (zorder=2)
if len(ceros) > 0:
    ax3.scatter(np.real(ceros), np.imag(ceros), s=80, facecolors='none', 
                edgecolors='blue', linewidth=2, label='Ceros (o)')

# Graficamos los Polos al frente de todo (zorder=3) y más grandes (s=120)
if len(polos) > 0:
    ax3.scatter(np.real(polos), np.imag(polos), s=120, marker='x', 
                color='orange', linewidth=3, label='Polos (x)')

# Ajustes estéticos para que sea idéntico al plano Z teórico
ax3.set_title('Mapa de Polos y Ceros (Plano Z)')
ax3.set_xlabel('$\\Re(z)$ (Parte Real)')
ax3.set_ylabel('$\\Im(z)$ (Parte Imaginaria)')
ax3.grid(True, which='both', linestyle=':', alpha=0.6, zorder=1)

# Forzamos la relación de aspecto 1:1 para que el círculo no se vea ovalado
ax3.axis('equal') 
ax3.set_xlim([-1.5, 1.5])
ax3.set_ylim([-1.5, 1.5])
ax3.legend(loc='upper left')

plt.show()
