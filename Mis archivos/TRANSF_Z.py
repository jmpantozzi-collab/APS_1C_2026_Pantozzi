#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from scipy.io.wavfile import write

#%% Ejemplo modulo y fase de orden 2
fs=500 #Tomamos algun valor por encima del doble de ws para que tenga sentido el sistema
wp = 70 #le puedo pasar sin normalizar porque la funcion normaliza directamente al pasarle fs
ws = 100
gpass = 1 #dB
gstop = 50 #dB
b_coeffs, a_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype='cheby1', output='ba')

taps=b_coeffs.shape[0]

w, h = sig.freqz(b_coeffs,a=a_coeffs,worN=1024, fs=fs)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 =axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
              #taps es la cantidad de terminos o orden -1
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
#ax1.plot(w, abs(h), 'C0')
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
ax1.set_ylabel("Mod en veces", color='C0')
ax1.grid(True)

phase = np.unwrap(np.angle(h)) #unwrap desenvuelve la respuesta de fase de 0 a pi
#phase = np.angle(h)

ax2.plot(w, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.set_xlabel("Frequency in rad/sample")
ax2.grid(True)
ax2.axis('tight')
plt.show()

#%% Pasa Banda
fs=500 #Tomamos algun valor por encima del doble de ws para que tenga sentido el sistema

#Armamos una plantilla de pasa banda para ECG de TS5 
wp1 = 1  
ws1 = .1
wp2 = 35 #ECG el bw esta al rededor de 30
ws2 = 45

wp=[wp1,wp2] #las empaquetamos como nos pide la funcion
ws=[ws1,ws2]
gpass = .5 #dB
gstop = 50 #dB

ftypes='butter'
#ftypes='cheby1'
#ftypes='cheby2'
#ftypes='cauer'

#b_coeffs, a_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype=ftypes, output='ba')
sos_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype=ftypes, output='sos')

taps=b_coeffs.shape[0]

w, h = sig.freqz(b_coeffs,a=a_coeffs,worN=1024, fs=fs)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 =axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
              #taps es la cantidad de terminos o orden -1
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
#ax1.plot(w, abs(h), 'C0')
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
ax1.set_ylabel("Mod en veces", color='C0')
ax1.grid(True)

phase = np.unwrap(np.angle(h)) #unwrap desenvuelve la respuesta de fase de 0 a pi
#phase = np.angle(h)

ax2.plot(w, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.set_xlabel("Frequency in rad/sample")
ax2.grid(True)
ax2.axis('tight')
plt.show()
