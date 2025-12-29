""" "
Plot results from an epop run
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
from astropy import units as u, constants as cst
import glob

"""
Visualize the outputs of a hierarchical histogram run
"""


# read in MCMC samples
n_msini_bins = 3
n_mass_bins = n_msini_bins
n_sma_bins = 1
n_e_bins = 5
n_burn = 500  # number of burn-in steps I ran for the actual MCMC
n_total = 500
nwalkers = 100
fullmarg = 'fullmarg_' # 'fullmarg_' or '' depending on model run
 
ndim = n_mass_bins * n_sma_bins * n_e_bins
savedir = f"../plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"

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

ecc_bin_edges = np.load("../completeness_model/{}ecc_bins.npy".format(n_e_bins))
sma_bin_edges = np.load("../completeness_model/{}sma_bins.npy".format(n_sma_bins))
msini_bin_egdes = np.load("../completeness_model/{}msini_bins.npy".format(n_msini_bins))
mass_bin_edges = msini_bin_egdes

d_logmsini = np.log(msini_bin_egdes[1:]) - np.log(msini_bin_egdes[:-1])
d_logmass = np.log(mass_bin_edges[1:]) - np.log(mass_bin_edges[:-1])
d_loga = np.log(sma_bin_edges[1:]) - np.log(sma_bin_edges[:-1])
assert np.all(np.isclose(d_loga, d_loga[0]))
d_loga = d_loga[0]

d_ecc = ecc_bin_edges[1:] - ecc_bin_edges[:-1]
assert np.all(np.isclose(d_ecc, d_ecc[0]))
d_ecc = d_ecc[0]

chains_flattened = chains.reshape((-1, n_e_bins, n_sma_bins, n_mass_bins))

"""
Plot total occurrence in the BD desert
"""

mass_idx = -1


dn_de_dloga = (
    chains_flattened[:, :, :, mass_idx] * d_logmass[mass_idx]
)  # (n_samples, e, a, m)

dn = 0
for i in range(n_e_bins):
    for j in range(n_sma_bins):
        dn += dn_de_dloga[:, i, j] * d_ecc * d_loga
bd_occurrence_rate = dn / nstars_cps

plt.figure()
plt.hist(bd_occurrence_rate, bins=50)

print(np.quantile(bd_occurrence_rate, [0.16, 0.5, 0.84]))
plt.savefig(f"{savedir}/bd_occurrence_rate_burn{n_burn}_total{n_total}.png", dpi=250)

print(savedir)
"""
compute significance of peak
"""


sma_bin_idx = 0
mass_bin_idx = 1

# NOTE: to directly compare absolute occurrence as below, the eccentricity bin sizes need to be equal
# (luckily they are, but I'd need to correct for unequal bin sizes if they weren't)

bin0_bin1_ratio = (
    chains_flattened[:, 1, sma_bin_idx, mass_bin_idx]
    / chains_flattened[:, 0, sma_bin_idx, mass_bin_idx]
)
bin1_bin2_ratio = (
    chains_flattened[:, 1, sma_bin_idx, mass_bin_idx]
    / chains_flattened[:, 2, sma_bin_idx, mass_bin_idx]
)

prob1 = bin0_bin1_ratio > 1  # prob that bin 1 hist is greater than bin 0
prob2 = bin1_bin2_ratio > 1  # prob that bin 1 hist is greater than bin 2

print(np.sum(prob1) / len(chains_flattened))
print(np.sum(prob2) / len(chains_flattened))

print(np.sum(prob2 & prob1) / len(chains_flattened))

quants = np.quantile(1 / bin0_bin1_ratio, [0.84, 0.95, 0.997])

# print(quants)

plt.figure()
plt.hist(
    1 / bin0_bin1_ratio, bins=50, color="grey", alpha=0.5, range=(0, 2), density=True
)
plt.axvline(1, color="purple")
plt.axvline(quants[0], color="grey", ls="-.", label="1$\\sigma$")
plt.axvline(quants[1], color="grey", ls="--", label="2$\\sigma$")
plt.axvline(quants[2], color="grey", ls="-", label="3$\\sigma$")
plt.legend()
plt.ylabel("N$_{{\\mathrm{{samples}}}}$")
plt.xlabel("$\\Gamma_{\\mathrm{{e}}\sim0.3}$ / $\\Gamma_{\\mathrm{{e}}\sim0.1}$")
plt.savefig(f"{savedir}/relativeprob_burn{n_burn}_total{n_total}.png", dpi=250)


"""
Occurrence samples plot
"""

fig = plt.figure(figsize=(5 * n_mass_bins, 5))

gs = fig.add_gridspec(1, n_mass_bins + 1, width_ratios=[20] * n_mass_bins + [1])
ax = []
for i in range(n_mass_bins):
    ax_i = fig.add_subplot(gs[0, i])

    ax.append(ax_i)
ax_cbar = fig.add_subplot(gs[0, -1])
ax.append(ax_cbar)

for a in ax[:-1]:
    a.set_xscale("log")
    a.set_xlim(sma_bin_edges[0], sma_bin_edges[-1])
    a.set_ylim(ecc_bin_edges[0], ecc_bin_edges[-1])
    a.set_xlabel("$a$ [au]")
ax[0].set_ylabel("$e$")
ax[1].set_ylabel("$e$")

for i, a in enumerate(ax[:-1]):
    a.set_title(
        "{:0.0f} M$_{{\\oplus}}$ < M < {:0.0f} M$_{{\\oplus}}$ \n {:.1f} M$_{{\\mathrm{{J}}}}$ < M < {:.1f} M$_{{\\mathrm{{J}}}}$ ".format(
            mass_bin_edges[i],
            mass_bin_edges[i + 1],
            (u.M_earth * mass_bin_edges[i]).to(u.M_jup).value,
            (u.M_earth * mass_bin_edges[i + 1]).to(u.M_jup).value,
        )
    )

for post_path in glob.glob("../lee_posteriors/resampled/ecc_*.csv"):

    ecc_post = pd.read_csv(post_path).values.flatten()
    post_len = len(ecc_post)

    st_name = post_path.split("/")[-1].split("_")[1]
    pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0]

    msini_post = pd.read_csv(
        f"../lee_posteriors/resampled/msini_{st_name}_{pl_num}.csv"
    ).values.flatten()
    sma_post = pd.read_csv(
        f"../lee_posteriors/resampled/sma_{st_name}_{pl_num}.csv"
    ).values.flatten()

    ax_idx = None
    for i in np.arange(n_mass_bins):
        if (
            mass_bin_edges[i] < np.median(msini_post)
            and np.median(msini_post) < mass_bin_edges[i + 1]
        ):
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
            color="grey",
        )

for i in np.arange(n_mass_bins):

    ax[i].tick_params(top=True, right=True, which="both", direction="in", length=3.5)

    dn_dmass_de_dloga = chains_flattened[:, :, :, i]  # (n_samples, e, a, m)
    dn_de_dloga = dn_dmass_de_dloga * d_logmass[i]

    dn = dn_de_dloga * d_ecc * d_loga

    total_occurrence = np.sum(np.sum(dn, axis=1), axis=1)

    # calculate the probability that a planet in this entire range of
    # msini/a lives in a specific e/sma/msini box
    d_prob = np.zeros((len(chains_flattened), n_e_bins, n_sma_bins))
    for j in range(n_e_bins):
        for k in range(n_sma_bins):
            d_prob[:, j, k] = dn[:, j, k] / total_occurrence * 100

    # sanity check: all the probabilities should add up to 100 for a given posterior sample
    assert np.isclose(np.max(np.sum(np.sum(d_prob, axis=1), axis=1)), 100)
    assert np.isclose(np.min(np.sum(np.sum(d_prob, axis=1), axis=1)), 100)

    medians = np.median(d_prob, axis=0)

    pc = ax[i].pcolormesh(
        sma_bin_edges,
        ecc_bin_edges,
        medians,
        shading="auto",
        cmap="Purples",
        vmin=0,
        vmax=75,
        edgecolor="white",
    )
    for j, a in enumerate(sma_bin_edges[:-1]):
        for k, e in enumerate(ecc_bin_edges[:-1]):

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

fig, ax = plt.subplots(1, n_mass_bins, figsize=(5 * n_mass_bins, 5))

sma_bin_idx = 0

# integrate over eccentricity and msini to get dN/d(lna)
n2plot = 100
idx2plot = np.random.choice(
    np.arange(len(chains_flattened[:, 0, 0, 0])),
    n2plot,
)
colors = ["rebeccapurple", "grey", "teal", "black"]
fmts = ["o", "^", "*", "*"]
for i in range(n_mass_bins):

    dn_dmass_de_dloga = chains_flattened[:, :, sma_bin_idx, i]  # (n_steps, n_e)
    dn = dn_dmass_de_dloga * d_logmass[i] * d_ecc * d_loga
    d_occurrence = dn / nstars_cps * 100

    for j, a in enumerate(ecc_bin_edges[:-1]):

        # each of these is dN/dMsini * de * dloga
        for k in range(n2plot):
            ax[i].plot(
                [ecc_bin_edges[j], ecc_bin_edges[j + 1]],
                np.ones(2) * d_occurrence[idx2plot[k], j],
                color=colors[i],
                alpha=0.1,
            )

        quantiles = np.quantile(d_occurrence[:, j], [0.16, 0.5, 0.84])
        ax[i].errorbar(
            [0.5 * (ecc_bin_edges[j] + ecc_bin_edges[j + 1])],
            [quantiles[1]],
            [[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]],
            color=colors[i],
            fmt=fmts[i],
        )

ax[0].set_ylabel("N$_{{\\mathrm{{pl}}}}$ / 100 stars")
ax[0].text(
    0,
    14,
    "a range: {:.1f}au < a < {:.1f}au".format(
        sma_bin_edges[sma_bin_idx], sma_bin_edges[sma_bin_idx + 1]
    ),
)
for i, a in enumerate(ax):
    a.set_xlabel("ecc")
    a.set_title(
        "{:0.0f} M$_{{\\oplus}}$ < M < {:0.0f} M$_{{\\oplus}}$ \n {:.1f} M$_{{\\mathrm{{J}}}}$ < M < {:.1f} M$_{{\\mathrm{{J}}}}$ ".format(
            mass_bin_edges[i],
            mass_bin_edges[i + 1],
            (u.M_earth * mass_bin_edges[i]).to(u.M_jup).value,
            (u.M_earth * mass_bin_edges[i + 1]).to(u.M_jup).value,
        )
    )
    a.set_ylim(0, 15)
plt.savefig(
    f"{savedir}/ecc_marginalized_samples_burn{n_burn}_total{n_total}_smaidx{sma_bin_idx}.png",
    dpi=250,
)

"""
corner plot
"""

corner.corner(chains_flattened)
plt.savefig(f"{savedir}/corner_burn{n_burn}_total{n_total}.png", dpi=250)

"""
trend plot
"""

fig, ax = plt.subplots(ndim, 1, figsize=(5, 30), sharex=True)
plt.subplots_adjust(hspace=0)

for i in range(nwalkers):
    for j in range(ndim):
        ax[j].plot(chains[:, i, j], alpha=0.05, color="k")

plt.savefig(f"{savedir}/trend_burn{n_burn}_total{n_total}.png", dpi=250)

print(savedir)
