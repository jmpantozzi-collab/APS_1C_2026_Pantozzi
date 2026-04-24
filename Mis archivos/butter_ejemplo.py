
#%% Inicialización e importación de módulos

# Módulos externos
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

fig_sz_x = 13
fig_sz_y = 7
fig_dpi = 80 # dpi

fig_font_size = 11

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi
plt.rcParams.update({'font.size':fig_font_size})


this_order=5
eps=1

z,p,k = sig.buttap(this_order)

num, den = sig.zpk2tf(z,p,k)
num, den = sig.lp2lp(num, den, eps**(-1/this_order))

z,p,k = sig.tf2zpk(num, den)

