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
n_mass_bins = 3
n_sma_bins = 2
n_e_bins = 4
n_burn = 500  # number of burn-in steps I ran for the actual MCMC
n_total = 500
nwalkers = 100

ndim = n_mass_bins * n_sma_bins * n_e_bins
savedir = f"plots/{n_mass_bins}msini{n_sma_bins}sma{n_e_bins}e"

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
msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_mass_bins))

d_logmsini = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
d_loga = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])

d_ecc = ecc_bins[1:] - ecc_bins[:-1]
assert np.all(np.isclose(d_ecc, d_ecc[0]))
d_ecc = d_ecc[0]

"""
compute significance of peak
"""

chains_flattened = chains.reshape((-1, n_e_bins, n_sma_bins, n_mass_bins))


sma_bin_idx = 1
msini_bin_idx = 2

# NOTE: to directly compare absolute occurrence as below, the bin sizes need to be equal
# (luckily they are, but I'd need to correct for unequal bin sizes if they weren't)

bin0_bin1_ratio = chains_flattened[:, 1, sma_bin_idx, msini_bin_idx]/chains_flattened[:, 0, sma_bin_idx, msini_bin_idx]
bin1_bin2_ratio = chains_flattened[:, 1, sma_bin_idx, msini_bin_idx]/chains_flattened[:, 2, sma_bin_idx, msini_bin_idx]

prob1  = (bin0_bin1_ratio > 1) # prob that bin 1 hist is greater than bin 0
prob2 = (bin1_bin2_ratio > 1) # prob that bin 1 hist is greater than bin 2

print(np.sum(prob1) / len(chains_flattened))
print(np.sum(prob2) / len(chains_flattened))

print(np.sum(prob2 & prob1) / len(chains_flattened))

quants = np.quantile(bin0_bin1_ratio, [0.05,0.16, 0.036])

print(quants)

plt.figure()
plt.hist(bin0_bin1_ratio, bins=50, color='grey',alpha=0.5,range=(0,20), density=True)
plt.axvline(1, color='purple')
plt.axvline(quants[0], color='grey', ls='--')
plt.axvline(quants[1], color='grey', ls='--')
plt.xlabel('sub-Jov prob. / sup-Jov prob.')
plt.savefig(f"{savedir}/relativeprob_burn{n_burn}_total{n_total}.png", dpi=250)


"""
Occurrence samples plot
"""

fig = plt.figure(figsize=(5*n_mass_bins, 5))
if n_mass_bins == 2:
    width_ratios=(20, 20, 1)
elif n_mass_bins == 3:
    width_ratios=(20, 20, 20, 1)
gs = fig.add_gridspec(1, n_mass_bins+1, width_ratios=width_ratios)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax_cbar = fig.add_subplot(gs[0, 3])
ax = [ax0, ax1, ax2, ax_cbar]


for a in ax[:-1]:
    a.set_xscale("log")
    a.set_xlim(sma_bins[0], sma_bins[-1])
    a.set_ylim(ecc_bins[0], ecc_bins[-1])
    a.set_xlabel("$a$ [au]")
ax[0].set_ylabel("$e$")
ax[1].set_ylabel("$e$")

for i, a in enumerate(ax[:-1]):
    a.set_title(
        "{:.1f} M$_{{\\oplus}}$ < M < {:.1f} M$_{{\\oplus}}$".format(
            msini_bins[i], msini_bins[i+1]
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

    ax_idx = None
    for i in np.arange(n_mass_bins):
        if msini_bins[i] < np.median(msini_post) and np.median(msini_post) < msini_bins[i+1]:
            ax_idx = i

    if ax_idx is not None:
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

for i in np.arange(n_mass_bins):

    ax[i].tick_params(top=True, right=True, which="both", direction="in", length=3.5)

    dn_dmsini_de_dloga = chains_flattened[:, :, :, i]
    dn_de_dloga = dn_dmsini_de_dloga * d_logmsini[i]
    
    dn = dn_de_dloga * d_ecc * d_loga

    total_occurrence = (
        np.sum(np.sum(dn, axis=1), axis=1)
    )

    # calculate the probability that a planet in this entire range of
    # msini/a lives in a specific e/sma/msini box
    d_prob = np.zeros((len(chains_flattened), n_e_bins, n_sma_bins))
    for j in range(n_e_bins):
        for k in range(n_sma_bins):
            d_prob[:,j,k] = dn[:,j,k] / total_occurrence * 100
    
    # sanity check: all the probabilities should add up to 100 for a given posterior sample
    assert np.isclose(np.max(np.sum(np.sum(d_prob, axis=1), axis=1)), 100)
    assert np.isclose(np.min(np.sum(np.sum(d_prob, axis=1), axis=1)),100)

    medians = np.median(d_prob, axis=0)

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

cbar = fig.colorbar(pc, cax=ax[-1], label="relative prob. (%)")
plt.savefig(f"{savedir}/samples_burn{n_burn}_total{n_total}.png", dpi=250)


"""
Marginalized 1d eccentricity plot for low vs high masses in a single sma bins
"""

fig, ax = plt.subplots(1, n_mass_bins, figsize=(5*n_mass_bins, 5))

sma_bin_idx = 1

# integrate over eccentricity and msini to get dN/d(lna)
n2plot = 100
idx2plot = np.random.choice(
    np.arange(len(chains_flattened[:, 0, 0, 0])),
    n2plot,
)
colors = [ "rebeccapurple", "grey","teal"]
fmts = ["o", "^", "*"]
for i in range(n_mass_bins):

    dn_dmsini_de_dloga = chains_flattened[:, :, sma_bin_idx, i]  # (n_steps, n_e)
    dn = dn_dmsini_de_dloga * d_logmsini[i] * d_ecc * d_loga[sma_bin_idx]
    d_occurrence = dn / nstars_cps * 100

    for j, a in enumerate(ecc_bins[:-1]):

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
        )

ax[1].set_ylabel("N$_{{\\mathrm{{pl}}}}$ / 100 stars")
ax[1].text(0,7.6, "a range: {:.1f}au < a < {:.1f}au".format(sma_bins[sma_bin_idx], sma_bins[sma_bin_idx+1]))
for i, a in enumerate(ax):
    a.set_xlabel("ecc")
    a.set_title("{:.1f} M$_{{\\oplus}}$ < M < {:.1f} M$_{{\\oplus}}$".format(
                msini_bins[i], msini_bins[i+1]
            )
)
    a.set_ylim(0,8)
plt.savefig(
    f"{savedir}/ecc_marginalized_samples_burn{n_burn}_total{n_total}_smaidx{sma_bin_idx}.png", dpi=250
)

"""
corner plot
"""

# corner.corner(chains)
# plt.savefig(f"{savedir}/corner_burn{n_burn}_total{n_total}.png", dpi=250)

"""
trend plot
"""

# fig, ax = plt.subplots(ndim, 1, figsize=(5, 30), sharex=True)
# plt.subplots_adjust(hspace=0)

# for i in range(nwalkers):
#     for j in range(ndim):
#         ax[j].plot(chains[:, i, j], alpha=0.05, color="k")

# plt.savefig(f"{savedir}/trend_burn{n_burn}_total{n_total}.png", dpi=250)
