import numpy as np
import matplotlib.pyplot as plt
import corner

"""
Draws values that are uniform in sqrt(e)sin(w)/sqrt(e)cos(w), converts them to e, 
then plots the result to show that it's uniform in e.

TODO: undo my sqrt(e) division in the HBM model
"""

n_samples = int(1e6)
s2esinw_samples = np.random.uniform(-1, 1, n_samples)
s2ecosw_samples = np.random.uniform(-1, 1, n_samples)

e_samples = s2esinw_samples**2 + s2ecosw_samples**2
omega_samples = np.degrees(np.arctan2(s2ecosw_samples, s2esinw_samples))

bins = 20

fig = corner.corner(
    np.transpose([e_samples, omega_samples]),
    labels=["ecc", "$\omega$ [deg]"],
    bins=bins,
)


fig.axes[0].hist(e_samples[e_samples < 1], color="rebeccapurple", bins=bins, alpha=0.5)


fig.axes[3].hist(
    omega_samples[e_samples < 1],
    color="rebeccapurple",
    alpha=0.5,
    bins=bins,
)

plt.savefig("plots/sqrt_e_samples.png", dpi=250)
