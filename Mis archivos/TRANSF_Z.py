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
gpass = 3 #dB
gstop = 10 #dB
b_coeffs, a_coeffs = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype='butter', output='ba')

taps=b_coeffs.shape[0]

w, h = sig.freqz(b_coeffs,a=a_coeffs,worN=1024, fs=fs)

fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
              #taps es la cantidad de terminos o orden -1
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
#ax1.plot(w, abs(h), 'C0')
ax1.set_ylabel("Mod en veces", color='C0')
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi))
ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h)) #unwrap desenvuelve la respuesta de fase de 0 a pi
#phase = np.angle(h)

ax2.plot(w, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

#%% irrdesing diseño de IIR
fs=500 #Tomamos algun valor por encima del doble de ws para que tenga sentido el sistema
wp = 70 #le puedo pasar sin normalizar porque la funcion normaliza directamente al pasarle fs
ws = 100
gpass = 3 #dB
gstop = 10 #dB
b_coeff, a_coeff = sig.iirdesign(wp, ws, gpass, gstop, fs=fs, analog=False, ftype='butter', output='ba')
w, h = sig.freqz(*system)
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid(True)
ax1.set_ylim([-120, 20])
ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h))
ax2.plot(w, phase, 'g')
ax2.set_ylabel('Phase [rad]', color='g')
ax2.grid(True)
ax2.axis('tight')
ax2.set_ylim([-6, 1])
nticks = 8
ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))