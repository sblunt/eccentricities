""" "
Plot results from an epop run
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
from astropy import units as u, constants as cst
import glob

# read in MCMC samples
n_msini_bins = 2
n_sma_bins = 2
n_e_bins = 5
n_burn = 500  # number of burn-in steps I ran for the actual MCMC
n_total = 500
nwalkers = 100

ndim = n_msini_bins * n_sma_bins * n_e_bins
savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"


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

ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))

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
Occurrence samples plot
"""

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 3, width_ratios=(20, 20, 1))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax = [ax0, ax1, ax2]

chains = chains.reshape((-1, n_e_bins, n_sma_bins, n_msini_bins))

for a in ax[:-1]:
    a.set_xscale("log")
    a.set_xlim(sma_bins[0], sma_bins[-1])
    a.set_ylim(ecc_bins[0], ecc_bins[-1])
    a.set_xlabel("sma [au]")
ax[0].set_ylabel("eccentricity")
ax[0].set_title(
    "{:.2f} M$_{{\\oplus}}$ < Msini < {:.2f} M$_{{\\oplus}}$".format(
        msini_bins[0], msini_bins[1]
    )
)
ax[1].set_title(
    "{:.2f} M$_{{\\oplus}}$ < Msini < {:.2f} M$_{{\\oplus}}$".format(
        msini_bins[1], msini_bins[2]
    )
)

for post_path in glob.glob("lee_posteriors/resampled/ecc_*.csv"):

    ecc_post = pd.read_csv(post_path).values.flatten()
    post_len = len(ecc_post)

    st_name = post_path.split("/")[-1].split("_")[1]
    pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0]

    msini_post = pd.read_csv(
        f"lee_posteriors/resampled/msini_{st_name}_{pl_num}.csv"
    ).values.flatten()
    sma_post = pd.read_csv(
        f"lee_posteriors/resampled/sma_{st_name}_{pl_num}.csv"
    ).values.flatten()

    if np.median(msini_post) > msini_bins[1]:
        ax_idx = 1
    else:
        ax_idx = 0

    ax[ax_idx].scatter(
        [np.median(sma_post)],
        [np.median(ecc_post)],
        color="white",
        ec="grey",
        zorder=10,
    )

    sma_cis = np.quantile(sma_post, [0.16, 0.5, 0.84])
    ecc_cis = np.quantile(ecc_post, [0.16, 0.5, 0.84])

    ax[ax_idx].errorbar(
        [sma_cis[1]],
        [ecc_cis[1]],
        xerr=([sma_cis[1] - sma_cis[0]], [sma_cis[2] - sma_cis[1]]),
        yerr=([ecc_cis[1] - ecc_cis[0]], [ecc_cis[2] - ecc_cis[1]]),
        color="white",
    )

plot_probability = True

for i in range(n_msini_bins):

    ax[i].tick_params(top=True, right=True, which="both", direction="in", length=3.5)

    dn_dmsini_de_dloga = chains[:, :, :, i]
    dn_de_dloga = dn_dmsini_de_dloga * d_logmsini[i]
    # these have dimensions (n_ecc_bins, n_sma_bins)
    # n here indicates total number of planets per bin that would
    # be detected by THIS survey. So to convert to occurrence rate,
    # we need to divide by the total number of stars in the survey

    d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100
    d_occurrence = d_occurrence_de_dloga * d_ecc * d_loga

    total_occurrence = (
        np.sum(np.sum(chains[:, :, :, i] * d_ecc, axis=1) * d_loga, axis=1)
        * d_logmsini[i]
    )
    # this is the total number of planets we expect to be in this msini range, detected or not
    print(np.median(total_occurrence))

    # calculate the probability that a planet in this entire range of
    # msini/a lives in a specific e/sma/msini box
    dn_de_dloga = np.array([dn_de_dloga[i] for i in range(len(total_occurrence))])

    d_prob = 100 * (dn_de_dloga * d_ecc * d_loga) / total_occurrence[i]

    if not plot_probability:
        medians = np.median(d_occurrence, axis=0)
        # stds = np.std(d_occurrence_de_dloga, axis=0)

        ax[i].pcolormesh(
            sma_bins,
            ecc_bins,
            medians,
            shading="auto",
            cmap="Purples",
            edgecolor="white",
            # vmin=0,
            # vmax=100,
        )
        for j, a in enumerate(sma_bins[:-1]):
            for k, e in enumerate(ecc_bins[:-1]):

                quantiles = np.quantile(d_occurrence[:, k, j], [0.16, 0.5, 0.84])

                # each of these is dN/dMsini * de * dloga
                ax[i].text(
                    a,
                    e + 0.02,
                    "{:.1f}$^{{+{:.1f}}}_{{-{:.1f}}}$ ".format(
                        medians[k, j],
                        quantiles[2] - quantiles[1],
                        quantiles[1] - quantiles[0],
                    ),
                    color="k",
                    zorder=20,
                )
    else:
        medians = np.median(d_prob, axis=0)
        # stds = np.std(d_occurrence_de_dloga, axis=0)

        pc = ax[i].pcolormesh(
            sma_bins,
            ecc_bins,
            medians,
            shading="auto",
            cmap="Purples",
            vmin=0,
            vmax=25,
            edgecolor="white",
        )
        for j, a in enumerate(sma_bins[:-1]):
            for k, e in enumerate(ecc_bins[:-1]):

                quantiles = np.quantile(d_prob[:, k, j], [0.16, 0.5, 0.84])

                # each of these is dN/dMsini * de * dloga
                ax[i].text(
                    a,
                    e + 0.02,
                    "{:.1f}$^{{+{:.1f}}}_{{-{:.1f}}}$ %".format(
                        medians[k, j],
                        quantiles[2] - quantiles[1],
                        quantiles[1] - quantiles[0],
                    ),
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
                    color="k",
                    zorder=20,
                )

cbar = fig.colorbar(pc, cax=ax[2], label="relative prob. (%)")
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

    dn_dmsini_de_dloga = chains[:, :, :, i]  # (n_steps, n_e, n_sma)

    dn_de_dloga = dn_dmsini_de_dloga * d_logmsini[i]
    d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100

    d_occurrence_dloga = np.sum(d_occurrence_de_dloga * d_ecc, axis=1)

    hist = []

    for j, a in enumerate(sma_bins[:-1]):

        label = None
        if j == 0 and i == 0:
            label = "{} Mearth < Msini < {} Mearth".format(msini_bins[0], msini_bins[1])
        elif j == 0 and i == 1:
            label = "{} Mearth < Msini < {} Mearth".format(msini_bins[1], msini_bins[2])

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
plt.savefig(f"{savedir}/marginalized_samples_burn{n_burn}_total{n_total}.png", dpi=250)


"""
Marginalized 1d eccentricity plot for low vs high masses in a single sma bins
"""

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sma_bin_idx = 1

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
    dn_dmsini_de_dloga = chains[:, :, sma_bin_idx:, i]  # (n_steps, n_e, n_sma)
    dn_de_dloga = dn_dmsini_de_dloga * d_logmsini[i]
    d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100

    d_occurrence_de = np.sum(d_occurrence_de_dloga * d_loga[sma_bin_idx:], axis=2)
    d_occurrence = d_occurrence_de * d_ecc

    for j, a in enumerate(ecc_bins[:-1]):

        label = None
        if j == 0 and i == 0:
            label = "{:.2f} M$_{{\\oplus}}$ < Msini < {:.2f} M$_{{\\oplus}}$".format(
                msini_bins[0], msini_bins[1]
            )
        elif j == 0 and i == 1:
            label = "{:.2f} M$_{{\\oplus}}$ < Msini < {:.2f} M$_{{\\oplus}}$".format(
                msini_bins[1], msini_bins[2]
            )

        # each of these is dN/dMsini * de * dloga
        for k in range(n2plot):
            ax[i].plot(
                [ecc_bins[j], ecc_bins[j + 1]],
                np.ones(2) * d_occurrence[idx2plot[k], j],
                color=colors[i],
                alpha=0.1,
            )

        quantiles = np.quantile(d_occurrence[:, j], [0.16, 0.5, 0.84])
        ax[i].errorbar(
            [0.5 * (ecc_bins[j] + ecc_bins[j + 1])],
            [quantiles[1]],
            [[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]],
            color=colors[i],
            fmt=fmts[i],
            label=label,
        )
for a in ax:
    a.set_xlabel("ecc")
    a.set_ylabel("N$_{{\\mathrm{{pl}}}}$ / 100 stars")
    a.legend()
    a.set_title("{:.2f}au < a < {:.2f}au".format(sma_bins[sma_bin_idx], sma_bins[-1]))
plt.savefig(
    f"{savedir}/ecc_marginalized_samples_burn{n_burn}_total{n_total}.png", dpi=250
)

"""
corner plot
"""

chains = chains.reshape((-1, ndim))

corner.corner(chains)
plt.savefig(f"{savedir}/corner_burn{n_burn}_total{n_total}.png", dpi=250)
