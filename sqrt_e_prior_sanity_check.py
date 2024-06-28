import numpy as np
import matplotlib.pyplot as plt

"""
Draws values that are uniform in sqrt(e), converts them to e, then plots the 
analytical prior p(e) propto e^-1/2 to check that these are the same. (They are.)
"""

sqrt_e_samples = np.random.uniform(0, 1, int(1e5))
nbins = 500
plt.hist(sqrt_e_samples, bins=nbins, label="sqrt(e)", alpha=0.5, density=True)
plt.hist(sqrt_e_samples**2, bins=nbins, label="e", alpha=0.5, density=True)

emin = 1 / nbins
x2plot = np.linspace(emin, 1, 100)

# normalization constant
A = 1 / (2 * (1 - np.sqrt(emin)))
plt.plot(x2plot, A / np.sqrt(x2plot))
plt.legend()
plt.savefig("plots/sqrt_e_samples.png", dpi=250)
