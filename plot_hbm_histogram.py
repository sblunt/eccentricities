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
n_burn = 500  # number of burn-in steps I ran for the actual MCMC
n_total = 200
nwalkers = 100

ndim = n_msini_bins * n_sma_bins * n_e_bins
savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"


nstars_cps = 719  # total number of stars in the sample

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

d_msini = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])

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
    # n here indicates total number of planets per bin that would
    # be detected by THIS survey. So to convert to occurrence rate,
    # we need to divide by the total number of stars in the survey

    dn_dmsini_de_dloga = chains[:, :, :, i]
    dn_de_dloga = dn_dmsini_de_dloga * d_msini[i]
    d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100

    medians = np.median(d_occurrence_de_dloga, axis=0)
    stds = np.std(d_occurrence_de_dloga, axis=0)

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
                "{:.1f}$\\pm${:.1f}".format(medians[k, j], stds[k, j]),
                color="white",
                zorder=20,
            )


plt.savefig(f"{savedir}/samples_burn{n_burn}_total{n_total}.png", dpi=250)


"""
Fulton+ 23 reproduction plot
Compare with https://content.cld.iop.org/journals/0067-0049/255/1/14/revision1/apjsabfcc1f5_hr.jpg
"""

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# integrate over eccentricity and msini to get dN/d(lna)
n2plot = 100
idx2plot = np.random.choice(
    np.arange(len(chains[:, 0, 0, 0])),
    n2plot,
)
colors = ["blue", "green"]
fmts = ["o", "^"]
for i in range(n_msini_bins):

    # for each posterior sample, add up delta_e * delta_msini * dN/de * d(loga) * d(msini) for each e bin
    dn_dmsini_de_dloga = chains[:, :, :, i]  # (n_steps, n_e, n_sma)
    dn_de_dloga = dn_dmsini_de_dloga * d_msini[i]
    d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100

    d_occurrence_dloga = np.sum(d_occurrence_de_dloga, axis=1) * d_ecc
    hist = []

    for j, a in enumerate(sma_bins[:-1]):

        label = None
        if j == 0 and i == 0:
            label = "{:.2f} Mj Msini < {:.2f} Mj".format(msini_bins[0], msini_bins[1])
        elif j == 0 and i == 1:
            label = "{:.2f} Mj < Msini < {:.2f} Mj".format(msini_bins[1], msini_bins[2])

        # each of these is dN/dMsini * de * dloga
        for k in range(n2plot):
            ax.plot(
                [sma_bins[j], sma_bins[j + 1]],
                np.ones(2) * d_occurrence_dloga[idx2plot[k], j],
                color=colors[i],
                alpha=0.1,
            )

        quantiles = np.quantile(d_occurrence_dloga[:, j], [0.16, 0.5, 0.84])
        ax.errorbar(
            [np.exp(0.5 * (np.log(sma_bins[j]) + np.log(sma_bins[j + 1])))],
            [quantiles[1]],
            [[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]],
            color=colors[i],
            fmt=fmts[i],
            label=label,
        )

ax.set_xscale("log")
ax.set_xlabel("sma [au]")
ax.set_ylim(0, 14)
ax.set_ylabel("N$_{{\\mathrm{{pl}}}}$ / 100 stars / $\\Delta$log(a)")
ax.legend()
# ax[0].set_title("{:.2f} Mj Msini < {:.2f} Mj".format(msini_bins[0], msini_bins[1]))
# ax[1].set_title("{:.2f} Mj < Msini < {:.2f} Mj".format(msini_bins[1], msini_bins[2]))
plt.savefig(f"{savedir}/marginalized_samples_burn{n_burn}_total{n_total}.png", dpi=250)

"""
corner plot
"""

chains = chains.reshape((-1, ndim))

corner.corner(chains)
plt.savefig(f"{savedir}/corner_burn{n_burn}_total{n_total}.png", dpi=250)
