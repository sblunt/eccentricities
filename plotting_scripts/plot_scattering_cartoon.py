from astropy import constants as cst, units as u
import numpy as np
import matplotlib.pyplot as plt

sma = np.logspace(-1, 1, int(1e2)) * u.au
Mst = 1 * u.M_sun
Rpl = 1 * u.R_jup


Mp_array = np.array([0.2, 0.3, 0.4])* u.M_jup

plt.figure(figsize=(5,5))

for i, Mp in enumerate(Mp_array):
    emax = np.sqrt(Mp * sma / (Rpl * Mst))
    plt.plot(sma, emax, alpha=(len(Mp_array)-i)/len(Mp_array), ls='--',label=f'{Mp.value} M$_{{\\mathrm{{Jup}}}}$', color='rebeccapurple')

plt.xlabel('a [au]')
plt.ylabel('e')
plt.legend()
plt.ylim(0,1)
plt.xlim(.1,10)
plt.xscale('log')
plt.savefig('../plots/scattering_cartoon.png',dpi=250)