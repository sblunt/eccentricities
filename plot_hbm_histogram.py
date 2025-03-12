""""
Plot results from an epop run
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd

# read in MCMC samples
n_msini_bins = 2
n_sma_bins = 6
n_e_bins = 1
n_burn = 200  # number of burn-in steps I ran for the actual MCMC
n_total = 200
nwalkers = 100

ndim = n_msini_bins * n_sma_bins * n_e_bins
savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"


# choose fraction of burn-in steps to chop off for plots
burnin = 0

posteriors = np.loadtxt(
    f"{savedir}/epop_samples_burn{n_burn}_total{n_total}.csv", delimiter=","
)
chains = posteriors.reshape((-1, nwalkers, ndim))

# remove burn-in
n_steps_perchain = len(chains)

chains = chains[int(burnin * n_steps_perchain) :, :, :]

ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))

d_msini = msini_bins[1:] - msini_bins[:-1]

d_ecc = ecc_bins[1:] - ecc_bins[:-1]
assert np.all(d_ecc == d_ecc[0])
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
Occurrence samples plot
"""

fig, ax = plt.subplots(1, n_msini_bins, figsize=(15, 5))

chains = chains.reshape((-1, n_e_bins, n_sma_bins, n_msini_bins))

for a in ax:
    a.set_xscale("log")
    a.set_xlim(sma_bins[0], sma_bins[-1])
    a.set_ylim(ecc_bins[0], ecc_bins[-1])
    a.set_xlabel("sma [au]")
    a.set_ylabel("eccentricity")
ax[0].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        msini_bins[0], msini_bins[1]
    )
)
ax[1].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        msini_bins[1], msini_bins[2]
    )
)

# overplot the planets
legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0, comment="#"
)
for i, row in legacy_planets.iterrows():

    # remove false positives
    if row.status not in ["A", "R", "N"]:

        if row.mass_med > msini_bins[1]:
            ax_idx = 1

        else:
            ax_idx = 0

        ax[ax_idx].scatter(
            [row.axis_med], [row.e_med], color="white", ec="grey", zorder=10
        )

        ax[ax_idx].errorbar(
            [row.axis_med],
            [row.e_med],
            xerr=([row.axis_med - row.axis_minus], [row.axis_plus - row.axis_med]),
            yerr=([row.e_med - row.e_minus], [row.e_plus - row.e_med]),
            color="white",
        )

for i in range(n_msini_bins):

    # these have dimensions (n_ecc_bins, n_sma_bins)
    medians = np.median(chains[:, :, :, i], axis=0)
    stds = np.std(chains[:, :, :, i], axis=0)

    ax[i].pcolormesh(
        sma_bins,
        ecc_bins,
        medians,
        shading="auto",
        # vmin=0,
        # vmax=100,
    )
    for j, a in enumerate(sma_bins[:-1]):
        for k, e in enumerate(ecc_bins[:-1]):

            # each of these is dN/dMsini * de * dloga
            ax[i].text(
                a,
                e,
                "{:.2f}$\\pm${:.2f}".format(medians[k, j], stds[k, j]),
                color="black",
                zorder=20,
            )

# NOTE: to compare directly to BJ, should use ~6 bins in log(sma) between 0.1 and 5 au
# try to reproduce https://content.cld.iop.org/journals/0067-0049/255/1/14/revision1/apjsabfcc1f5_hr.jpg

plt.savefig(f"{savedir}/samples_burn{n_burn}_total{n_total}.png", dpi=250)


# TODO: check this so that it's integrating over log(sma) and log(msini)
# fig, ax = plt.subplots(1, n_msini_bins, figsize=(15, 5))

# # integrate over eccentricity and msini to get dN/d(lna)
# for i in range(n_msini_bins):

#     # for each posterior sample, add up delta_e * delta_msini * dN/de * d(loga) * d(msini) for each e bin
#     dN_dlna = (
#         np.sum(chains[:, :, :, i], axis=1) * d_ecc * d_msini[i]
#     )  # (nsteps, e, sma, msini)
#     for j, a in enumerate(sma_bins[:-1]):

#         # each of these is dN/dMsini * de * dloga
#         for k in range(len(dN_dlna[:, j])):
#             ax[i].plot(
#                 [sma_bins[j], sma_bins[j + 1]],
#                 np.ones(2) * dN_dlna[k, j],
#                 color="k",
#                 alpha=0.01,
#             )
# for a in ax:
#     a.set_xscale("log")
#     a.set_xlabel("sma [au]")
#     a.set_ylabel("dN/d(loga)")
# ax[0].set_title("Msini < {} Mj".format(msini_bins[1]))
# ax[1].set_title("{} Mj < Msini < {} Mj".format(msini_bins[1], msini_bins[2]))
# plt.savefig(f"{savedir}/marginalized_samples_burn{n_burn}_total{n_total}.png", dpi=250)

# """
# corner plot
# """
# print(chains.shape)

# chains = chains.reshape((-1, ndim))

# print(chains.shape)

# corner.corner(chains)
# plt.savefig(f"{savedir}/corner_burn{n_burn}_total{n_total}.png", dpi=250)
