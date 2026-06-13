# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:16:16 2026

@author: jerem
"""

#%%  Diseño de FIRs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig 
import scipy.io as sio  


'-----DISEÑO FILTRO FIR REMEZ----'
fs_ecg = 1000
mat_struct = sio.loadmat('./TS5/ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()

fs = 1000
ripple = 1 

ws1 = .2 #STOP
wp1 = 1.2 #PASO
 
ws2 = 36.5 #STOP
wp2 = 35 # con 99% de energia me dio 31.41 

gpass = 1
gstop = 40

wp = [wp1, wp2]
ws = [ws1, ws2]

numtaps = 1721  #el orden de mi filtro es ceficientes -1
# tipo 2, coeficientes par, retardo no entero 
#numtaps = 31 # tipo 1, coeficientes impar, retardo entero

demora = (numtaps-1)//2 # // para que sea entero 

#gains = 10**((-1)*np.array([gstop, gstop, gpass, gpass, gstop, gstop])/20)
gains = np.array([0,1, 0])

if numtaps % 2 == 0:
    gains[-1] = 0. 
    
b_win = sig.remez(numtaps, bands= np.array([0. , ws1, wp1, wp2, ws2, fs//2]),
                  type='hilbert',
                  desired=gains,
                  weight=([7,1,25]),
                  fs=fs)
#el peso "weight" debe ser un array con la mitad de terminos que el nads (agarro el bands y cada 2 puntos tengo 1 region)


'RETARDO DE GRUPO'
ww = np.concatenate([
    np.logspace(start=-2, stop=0.1, num=500), 
    np.linspace(start=1.26, stop=35, num=200),
    np.logspace(start=1.55, stop=1.65, num=300),
    np.linspace(start=46, stop=fs//2, num=50)]) 


zeros, poles, gain = sig.tf2zpk(b_win,a = 1)

# 2. Calculamos la respuesta en frecuencia para el retardo de grupo
w, h = sig.freqz(b_win, worN=ww, fs=fs) # w son las frecuencias, H es la respuesta compleja

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
ax1.set_xlim([-10, 100])  # Acercamos el zoom para ver en detalle el ECG (hasta 0.3 de Nyquist = 75Hz)
ax1.set_ylim([-100, 5])
ax1.grid(True, which='both', linestyle='-', alpha=0.5)
ax1.legend(loc='lower left')

fase = np.unwrap(np.angle(h))

#%%
retardo_grupo = -np.diff(fase) / np.diff(ww*2*np.pi/(fs))
w_gd = w[:-1] # Ajustamos el tamaño del eje X por la derivada

# --- GRÁFICO 1: RETARDO DE GRUPO ---
'RETARDO DE GRUPO'
fig3, ax_gd = plt.subplots(figsize=(8, 4))
ax_gd.plot(w_gd, retardo_grupo, 'C2', linewidth=2, label='Retardo Real')
ax_gd.set_title("Retardo de Grupo del Filtro Pasa-Banda FIR")
ax_gd.set_xlabel("Frecuencia [Hz]")
ax_gd.set_ylabel("Retardo [muestras]")

#ax_gd.axvspan(wp1, wp2, color='gray', alpha=0.2, label='Banda de Paso (1-35 Hz)')
ax_gd.grid(True, linestyle=':', alpha=0.7)
ax_gd.legend()
ax_gd.set_xlim([0, fs/2])  
ax_gd.set_ylim([0, 1.1*np.max(retardo_grupo)])  
plt.tight_layout()

# --- GRÁFICO 2: DIAGRAMA DE POLOS Y CEROS ---
'POLOS Y CEROS'
fig4, ax_z = plt.subplots(figsize=(6, 6))
unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax_z.add_patch(unit_circle)
ax_z.axhline(0, color='gray', linewidth=1)
ax_z.axvline(0, color='gray', linewidth=1)

ax_z.scatter(np.real(zeros), np.imag(zeros), s=60, marker='o', facecolors='none', edgecolors='b', label='Ceros')
ax_z.scatter(np.real(poles), np.imag(poles), s=90, marker='x', color='r', label='Polos')

ax_z.set_aspect('equal')
ax_z.set_xlim([-1.2, 1.2])
ax_z.set_ylim([-1.2, 1.2])
ax_z.set_xlabel('Parte Real')
ax_z.set_ylabel('Parte Imaginaria')
ax_z.set_title('Diagrama de Polos y Ceros (Filtro Pasa-Banda) FIR')
ax_z.legend(loc='upper right')
ax_z.grid(True, linestyle=':', alpha=0.7)

plt.show()

'PLANTILLA Y CURVA DEL FILTRO'
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np


#%% ==========================================
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


sos_coeff_butt = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='butter', output='sos', fs=fs)
sos_coeff_cauer = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='cauer', output='sos', fs=fs)
sos_coeff_cheby1 = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='cheby1', output='sos', fs=fs)
sos_coeff_cheby2 = sig.iirdesign(wp=wp, ws=ws, gpass=gpass/2, gstop=gstop/2, analog=False, ftype='cheby2', output='sos', fs=fs)

#yy = sig.sosfilt(sos_coeff, ecg_one_lead)
yy_butt = sig.sosfiltfilt(sos_coeff_butt, ecg_one_lead)
yy_cauer = sig.sosfiltfilt(sos_coeff_cauer, ecg_one_lead)
yy_cheby1 = sig.sosfiltfilt(sos_coeff_cheby1, ecg_one_lead)
yy_cheby2 = sig.sosfiltfilt(sos_coeff_cheby2, ecg_one_lead)

cant_muestras = len(ecg_one_lead)
ecg_one_lead = ecg_one_lead

ECG_f_butt = yy_butt
ECG_f_cauer = yy_cauer
ECG_f_cheby1 = yy_cheby1
ECG_f_cheby2 = yy_cheby2

ECG_f_win = sig.lfilter(b_win, 1., ecg_one_lead)
#se podria ajustar un poco mas lo de bajo frecuencia porque hay un salto muy grande sin ruido en las frecuencias bajas 
#BAJARLE UN POQUITO LA FRECUENCIA INFERIOR, NO ASI DONDE ESTA LA PARTE DE ALTA FRECUENCIA
#NOS DIMOS CUENTA QUE LA PLANTILLA NO ESTABA TAN BIEN PORQUE NO COINCIDIA 

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
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth', linewidth=1)
    # plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer',  linewidth=1)
    # plt.plot(zoom_region, ECG_f_cheby1[zoom_region], label='Cheby1',  linewidth=1)
    # plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2', linewidth=1)
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
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
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth', linewidth=1)
    # plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer', linewidth=1)
    # plt.plot(zoom_region, ECG_f_cheby1[zoom_region], label='Cheby1', linewidth=1)
    # plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2', linewidth=1)
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG CON RUIDO filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

#%%



# firwin2 devuelve los coeficientes b del filtro fir de fase lineal (uno de los cuatro tipos), los a es solo 1 el primero el resto es 0
# no hay ai entonces no hay recursiones largas, no necesito dividirlos en varios filtros sos 
# firwin1 solo hace tipo 1 y tipo 2, no hace antisimetricos 

#iir son mas eficientes porque aprovechan los ai que los fir desde el punto computancional, por lo tanto orden 2 en fir es menor que las peores de las fir 

#en el diagrama de polos y ceros vemos como si tironean para arriba y para abajo (para gains no 1 y ceros)

#HEREDO ATENUACION DE FLATTOP PERO UNICA FORMA DE TRANSICIONAR RAPIDO ES UNA QUE TENGA UNA RAPIDA ATENUACION 
#la que mejor lo hace es la rectangular porque hace una rapida atenuacion 
#ahora que aplique la rectangular voy subiendo el numtaps (orden) para ver si puede cumplir el filtro

#en el -infinito en el tipo dos se nota que se va al cero en nyquist 
#nos esta apretando el zapato la primera banda de stop, para eso sirven los vectores de peso y por eso los otros metodos son mas sofisticados, en este no podemos hacer nada 

#como forzamos que sea de tipo 1, con 3601 me aseguro un retardo ENTERO (3601-3600/2) (no puedo poner 3600 en mi spyder porque se rompe)
#el diagrama de polos y ceros pasa a no tener informacion porque hay ceros por doquier, quizas en la banda de paso es lo mas interesante pero nada mas que eso 
#el diseño es mas que nada exigente por la primera parte. nos quedamos con numtaps = 1800
#misma performance que el butter para la señal que tiene ruido y es super costoso computacionalmente, un poquito peor que el butter
#posiblemente se deba porque no llegan bien a la atenuacion deseada 

#posibles tareas -> mejorar el diseño sin romper las maquinas
#ver si podemos lograr que el filtro logre llegar a la respuesta deseada
#probar firls