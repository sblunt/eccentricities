import numpy as np
import matplotlib.pyplot as plt
import corner
import astropy.units as u, astropy.constants as cst

"""
Draws values that are uniform in sqrt(e)sin(w)/sqrt(e)cos(w), converts them to e, 
then plots the result to show that it's uniform in e.
"""

# n_samples = int(1e4)
# s2esinw_samples = np.random.uniform(-1, 1, n_samples)
# s2ecosw_samples = np.random.uniform(-1, 1, n_samples)

# e_samples = s2esinw_samples**2 + s2ecosw_samples**2
# omega_samples = np.degrees(np.arctan2(s2ecosw_samples, s2esinw_samples))

# bins = 20

# fig = corner.corner(
#     np.transpose([e_samples, omega_samples]),
#     labels=["ecc", "$\\omega$ [deg]"],
#     bins=bins,
# )


# fig.axes[0].hist(e_samples[e_samples < 1], color="rebeccapurple", bins=bins, alpha=0.5)


# fig.axes[3].hist(
#     omega_samples[e_samples < 1],
#     color="rebeccapurple",
#     alpha=0.5,
#     bins=bins,
# )

# plt.savefig("plots/sanity_checks/sqrt_e_samples.png", dpi=250)

# """
# Converts known msini to mass, accounting for unknown inclination
# """

# Msini = 5
# inc = np.arccos(np.random.uniform(-1, 1, int(1e5)))
# M = Msini / np.sin(inc)

# plt.figure()
# plt.hist(M, bins=50, range=(0, 10))
# plt.savefig("plots/sanity_checks/m_vs_msini.png", dpi=250)

"""
Checks how uniformly sampling in P translates to sma
"""

n_samples = int(1e4)
Pmax = 20
P = np.random.uniform(0, Pmax, size=n_samples)  # [yrs]
M = np.random.normal(1, 0.05, size=n_samples)  # [M_sol]
sma = (P**2 * M) ** (1 / 3)

sma2plot = np.linspace(0, 7, int(1e2))

# B = (5**3 / (3**3 * M)) ** (1 / 5)
# normC = 8 / (5 * B * Pmax ** (8 / 5))


plt.hist(sma, bins=100, density=True, label="radvel prior")
# plt.plot(sma2plot, 5 * normC * B * sma2plot ** (3 / 5))
plt.plot(sma2plot, 0.07 * sma2plot ** (3 / 5), label="a^3/5")

plt.xlabel("sma [au]")
plt.ylabel("prob")
plt.legend()
plt.savefig("plots/sanity_checks/PvsSma.png", dpi=250)

"""
Checks how uniformly sampling in K translates to Msini
"""

P = np.random.uniform(0, Pmax, size=n_samples)  # [yrs]
Mst = np.random.normal(1, 0.05, size=n_samples)  # [M_sol]
e = np.random.uniform(0, 1, size=n_samples)
K = np.random.uniform(0, 100, size=n_samples)  # [m/s]


msini = (
    (2 * np.pi * cst.G / (P * u.yr)) ** (-1 / 3)
    * (Mst * u.M_sun) ** (2 / 3)
    * np.sqrt(1 - e**2)
) * (K * u.m / u.s)

plt.figure()
plt.hist(msini.to(u.M_jup), bins=50)
plt.savefig("plots/sanity_checks/KMsini.png", dpi=250)
