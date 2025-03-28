import numpy as np
import matplotlib.pyplot as plt
import corner
from orbitize.priors import LinearPrior

"""
Draws values from a linearly increasing prior, then importance samples
them to be consistent with a linearly decreasing prior
"""

n_samples = int(1e5)
oldLinearPrior = LinearPrior(-0.5, 1.5)
oldLinearPrior_norm = -0.5 * oldLinearPrior.b**2 / oldLinearPrior.m
newLinearPrior = LinearPrior(-2, 4.0)
newLinearPrior_norm = -0.5 * newLinearPrior.b**2 / newLinearPrior.m
samples = oldLinearPrior.draw_samples(n_samples)

old_prior_probs = np.exp(oldLinearPrior.compute_lnprob(samples))
new_prior_probs = np.exp(newLinearPrior.compute_lnprob(samples))
importance_weights = new_prior_probs / old_prior_probs
importance_probs = importance_weights / np.sum(importance_weights)

resampled_samples = np.random.choice(samples, size=int(1e5), p=importance_probs)

x2plot = np.array([0, 5])
old_prior_pred = (oldLinearPrior.m * x2plot + oldLinearPrior.b) / oldLinearPrior_norm
new_prior_pred = (newLinearPrior.m * x2plot + newLinearPrior.b) / newLinearPrior_norm
plt.figure()
plt.hist(
    samples,
    bins=50,
    density=True,
    alpha=0.5,
    color="purple",
    label="original samples",
)
plt.plot(x2plot, old_prior_pred, color="purple", label="original prior")
plt.hist(
    resampled_samples,
    bins=50,
    density=True,
    alpha=0.5,
    color="grey",
    label="importance resampled",
)
plt.plot(x2plot, new_prior_pred, color="k", label="new prior")
plt.ylim(0, 1)
plt.legend()
plt.savefig("plots/sanity_checks/prior_resampling.png", dpi=250)


"""
P is drawn from a uniform distribution. This checks the distribution of sma (assuming
stellar mass is a constant).

https://math.stackexchange.com/questions/2842360/if-x-has-a-u-1-3-distribution-and-y-x2-find-the-probability-density
"""

Pmin = 0
Pmax = 10
P = np.random.uniform(0, Pmax, int(1e5))
sma = P ** (2 / 3)

sma_min = 0
sma_max = Pmax ** (2 / 3)


norm_const = (2 / 3 * sma_max ** (3 / 2)) ** (-1)

plt.figure()
plt.hist(sma, bins=50, density=True)
sma2plot = np.linspace(sma_min, sma_max)
plt.plot(sma2plot, norm_const * sma2plot ** (1 / 2))
plt.savefig("plots/sanity_checks/P_vs_sma.png", dpi=250)


"""
Draws values that are uniform in sqrt(e)sin(w)/sqrt(e)cos(w), converts them to e, 
then plots the result to show that it's uniform in e.
"""

n_samples = int(1e5)
s2esinw_samples = np.random.uniform(-1, 1, n_samples)
s2ecosw_samples = np.random.uniform(-1, 1, n_samples)

e_samples = s2esinw_samples**2 + s2ecosw_samples**2
omega_samples = np.degrees(np.arctan2(s2ecosw_samples, s2esinw_samples))

bins = 20

fig = corner.corner(
    np.transpose([e_samples, omega_samples]),
    labels=["ecc", "$\\omega$ [deg]"],
    bins=bins,
)


fig.axes[0].hist(e_samples[e_samples < 1], color="rebeccapurple", bins=bins, alpha=0.5)


fig.axes[3].hist(
    omega_samples[e_samples < 1],
    color="rebeccapurple",
    alpha=0.5,
    bins=bins,
)

plt.savefig("plots/sanity_checks/sqrt_e_samples.png", dpi=250)

"""
Checks how uniformly sampling in e translates to sqrt(1-e^2)
"""

n_samples = int(1e4)
e = np.random.uniform(0, 1, size=n_samples)

e2plot = np.linspace(0, 1, 100, endpoint=False)

plt.figure()
plt.hist(np.sqrt(1 - e**2), bins=50, density=True)
plt.plot(e2plot, e2plot / np.sqrt(1 - e2plot**2))
plt.savefig("plots/sanity_checks/sqrt1minusesq.png", dpi=250)

"""
Checks how uniformly sampling in P translates to P^(1/3)
"""

n_samples = int(1e4)
Pmax = 30
P = np.random.uniform(0, Pmax, size=n_samples)

P2plot = np.linspace(0, Pmax ** (1 / 3), 100, endpoint=False)
norm_const = 3 / Pmax

plt.figure()
plt.hist(P ** (1 / 3), bins=50, density=True)
plt.plot(P2plot, norm_const * P2plot**2)
plt.savefig("plots/sanity_checks/P13.png", dpi=250)

"""
Checks how uniformly sampling in P and e translates to P^(1/3)sqrt(1-e^2)
https://math.stackexchange.com/questions/55684/pdf-of-product-of-variables
"""

np.random.seed(1)

n_samples = int(1e5)
Pmax = 30
Kmax = 100
P = np.random.uniform(0, Pmax, size=n_samples)
e = np.random.uniform(0, 1, size=n_samples)

z2plot = np.linspace(0, Pmax ** (1 / 3), 100)
norm_const = 3 / Pmax

plt.figure()
plt.hist(np.sqrt(1 - e**2) * P ** (1 / 3), bins=50, density=True)

plt.plot(
    z2plot, 3 * z2plot / Pmax ** (2 / 3) * np.sqrt(1 - z2plot**2 * Pmax ** (-2 / 3))
)
plt.savefig("plots/sanity_checks/Pe.png", dpi=250)

"""
Checks how uniformly sampling in P, e, and K translates to P^(1/3)sqrt(1-e^2)K
and to Msini (which is the same expression as above, with K in multiples of 
28.4329 m/s (Jupiter around Earth), P in years, and all multiplied by Mtot^(2/3), 
with Mtot in Msol)
"""


n_samples = int(1e5)
Pmax = 4  # [units of yr]
Kmax_ms = 10_000  # [m/s]
st_mass = 0.7
Kmax = Kmax_ms / 28.4329 * st_mass ** (2 / 3)
print(Kmax)
# Kmax = 100  # [units of Jupiter semiamplitude times (stellar mass)^(2/3) (i.e. 1 means Mst^(2/3) * 28.4329 m/s)]
P = np.random.uniform(0, Pmax, size=n_samples)  # [yr]
e = np.random.uniform(0, 1, size=n_samples)  # []
K = np.random.uniform(0, Kmax, size=n_samples)

plt.figure()
plt.hist(K * np.sqrt(1 - e**2) * P ** (1 / 3), bins=50, density=True)

plotme = np.linspace(0, Pmax ** (1 / 3) * Kmax, int(1e3))
print(np.max(plotme))


M = Kmax * Pmax ** (2 / 3)
A = Pmax ** (-2 / 3)

print(M)
print(A)


def expr(x, A):
    term1 = 1 / (2 * np.sqrt(A)) * np.arcsin(np.sqrt(A) * x)
    term2 = x / 2 * np.sqrt(1 - (A * x**2))
    return term1 + term2


pdf = 3 / M * (expr(Pmax ** (1 / 3), A) - expr(plotme / Kmax, A))

print(plotme)

plt.plot(plotme, pdf)
plt.savefig("plots/sanity_checks/PeK.png", dpi=250)
