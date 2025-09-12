import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
from astropy import units as u, constants as cst
import glob

# read in MCMC samples
n_msini_bins = 2
n_sma_bins = 1
n_e_bins = 5
n_burn = 10  # number of burn-in steps I ran for the actual MCMC
n_total = 50
nwalkers = 100

ndim = n_msini_bins * 3
savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e_parab"

nstars_cps = 719  # total number of stars in the sample

# choose fraction of burn-in steps to chop off for plots
burnin = 0  # 0.75

posteriors = np.loadtxt(
    f"{savedir}/epop_samples_burn{n_burn}_total{n_total}.csv", delimiter=","
)
chains = posteriors.reshape((-1, nwalkers, ndim))

# remove burn-in
n_steps_perchain = len(chains)

chains = chains[int(burnin * n_steps_perchain) :, :, :]

ecc_bins = np.load("completeness_model/{}ecc_bins_highsmaonlyTrue.npy".format(n_e_bins))
sma_bins = np.load(
    "completeness_model/{}sma_bins_highsmaonlyTrue.npy".format(n_sma_bins)
)
msini_bins = np.load(
    "completeness_model/{}msini_bins_highsmaonlyTrue.npy".format(n_msini_bins)
)

d_logmsini = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
d_loga = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])

d_ecc = ecc_bins[1:] - ecc_bins[:-1]
assert np.all(np.isclose(d_ecc, d_ecc[0]))
d_ecc = d_ecc[0]

"""
trend plot
"""


fig, ax = plt.subplots(ndim, 1, figsize=(5, 30), sharex=True)
plt.subplots_adjust(hspace=0)


for i in range(nwalkers):
    for j in range(ndim):
        ax[j].plot(chains[:, i, j], alpha=0.05, color="k")

plt.savefig(f"{savedir}/trend_burn{n_burn}_total{n_total}.png", dpi=250)

"""
samples plot
"""

chains = chains.reshape((-1, ndim))

n2plot = 100
rand_idx = np.random.choice(np.arange(len(chains[:, 0])), n2plot)

fig, ax = plt.subplots(1, n_msini_bins, figsize=(10, 5))

e2plot = np.linspace(0, 1, int(1e2))

for i, a in enumerate(ax):
    A_samples = chains[rand_idx, 3 * i]
    B_samples = chains[rand_idx, 3 * i + 1]
    C_samples = chains[rand_idx, 3 * i + 2]

    for j in range(n2plot):
        a.plot(
            e2plot,
            (A_samples[j] * e2plot**2 + B_samples[j] * e2plot + C_samples[j]),
            color="grey",
            alpha=0.5,
        )

    a.set_title(
        "{:d} M$_{{\\oplus}}$ < Msini < {:d} M$_{{\\oplus}}$".format(
            msini_bins[i], msini_bins[i + 1]
        )
    )
    a.set_xlabel("ecc")
    a.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{savedir}/samples_burn{n_burn}_total{n_total}.png", dpi=250)
