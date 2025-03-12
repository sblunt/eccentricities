"""
Make a plot to compare with Fig 2a of https://arxiv.org/pdf/1906.03266
that takes completeness into account

NOTE: I divided the samples by msini, not absolute mass, in model and data cases
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import copy
from astropy import units as u
from scipy.stats import gaussian_kde

n_ecc_bins = 1
n_sma_bins = 6
n_mass_bins = 2  # [<1.17 Mj and > 1.17Mj is binning used in Frelikh+]

# NOTE: SAMPLE SELECTION DEFINED HERE (should always be tighter limits than in get_posteriors.py)
ecc = np.linspace(0, 1, n_ecc_bins + 1)
sma = np.logspace(np.log10(0.1), np.log10(4), n_sma_bins + 1)
mass = np.array([0.1, 1, 13])  # [Mj]

recoveries = np.zeros((n_ecc_bins, n_sma_bins, n_mass_bins))
injections = np.zeros((n_ecc_bins, n_sma_bins, n_mass_bins))

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files:
    df = pd.read_csv(f)

    df["ecc_completeness_bins"] = np.nan
    df["sma_completeness_bins"] = np.nan
    df["mass_completeness_bins"] = np.nan
    for i in np.arange(len(ecc) - 1):
        df["ecc_completeness_bins"][
            ((df.inj_e.values >= ecc[i]) & (df.inj_e.values < ecc[i + 1]))
        ] = i
    for i in np.arange(len(sma) - 1):
        df["sma_completeness_bins"][
            ((df.inj_au.values >= sma[i]) & (df.inj_au.values < sma[i + 1]))
        ] = i

    for i in np.arange(len(mass) - 1):
        df["mass_completeness_bins"][
            (
                (df.inj_msini.values * (u.M_earth / u.M_jup).to("") >= mass[i])
                & (df.inj_msini.values * (u.M_earth / u.M_jup).to("") < mass[i + 1])
            )
        ] = i

    recovered_planets = df[df.recovered.values]
    unrecovered_planets = df[~df.recovered.values]

    for i, row in unrecovered_planets.iterrows():
        # check if injected planet is in range
        if not np.isnan(
            row.ecc_completeness_bins
            + row.sma_completeness_bins
            + row.mass_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.mass_completeness_bins),
            ] += 1

    for i, row in recovered_planets.iterrows():
        # check if injected planet is in range
        if not np.isnan(
            row.ecc_completeness_bins
            + row.sma_completeness_bins
            + row.mass_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.mass_completeness_bins),
            ] += 1
            recoveries[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.mass_completeness_bins),
            ] += 1

# compute completeness
completeness = recoveries / injections

# values in the 3d grid where there are 0 injections or 0 recoveries
bad_mask = (completeness == 0) | (np.isnan(completeness))

print(
    "Need to interpolate {} values ({:.2f}%)".format(
        np.sum(bad_mask), np.sum(bad_mask) / len(completeness.flatten()) * 100
    )
)

bad_idx = np.where(bad_mask)

# get (e,sma,msini) coords where we need to interpolate values
iterp_here = []

for idx, i in enumerate(bad_idx[0]):

    iterp_here.append([i, bad_idx[1][idx], bad_idx[2][idx]])

filled_in_points = scipy.interpolate.interpn(
    (ecc[:-1], sma[:-1], mass[:-1]),
    np.ma.array(completeness, mask=bad_mask),
    np.array(iterp_here),
    bounds_error=False,
    fill_value=0,
)

completeness_model = copy.copy(completeness)
completeness_model[bad_mask] = filled_in_points

# save the completeness model
np.save(
    "completeness_model/{}{}{}completeness".format(n_mass_bins, n_ecc_bins, n_sma_bins),
    completeness_model,
)
np.save("completeness_model/{}msini_bins".format(n_mass_bins), mass)
np.save("completeness_model/{}ecc_bins".format(n_ecc_bins), ecc)
np.save("completeness_model/{}sma_bins".format(n_sma_bins), sma)

completeness_model_lowmass = completeness_model[:, :, 0]
completeness_model_himass = completeness_model[:, :, 1]


"""
COMPLETENESS PLOT 
"""

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 3, width_ratios=(20, 20, 1))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax = [ax0, ax1, ax2]

ax[0].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[0], mass[1]
    )
)
ax[1].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[1], mass[2]
    )
)

ax[0].pcolormesh(sma, ecc, completeness_model_lowmass, shading="auto", vmin=0, vmax=1)
pc = ax[1].pcolormesh(
    sma, ecc, completeness_model_himass, shading="auto", vmin=0, vmax=1
)
cbar = fig.colorbar(pc, cax=ax[2])
cbar.set_label("completeness")

for a in ax[:2]:
    a.set_xscale("log")
    a.set_xlim(sma[0], sma[-1])
    a.set_ylim(0, 1)
    a.set_xlabel("$a$ [au]")
    a.set_ylabel("$e$")


# overplot planet params from CLS
legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0, comment="#"
)

lowmass_eccs = []
lowmass_smas = []
highmass_eccs = []
highmass_smas = []
all_masses = []
lowmass_ecc_errors = []
highmass_ecc_errors = []

for i, row in legacy_planets.iterrows():

    # remove false positives
    if row.status not in ["A", "R", "N"]:
        all_masses.append(row.mass_med)

        if row.mass_med > mass[1]:
            ax_idx = 1
            highmass_eccs.append(row.e_med)
            highmass_smas.append(row.axis_med)
            highmass_ecc_errors.append(
                np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            )

        else:
            ax_idx = 0
            lowmass_eccs.append(row.e_med)
            lowmass_smas.append(row.axis_med)
            lowmass_ecc_errors.append(
                np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            )

        ax[ax_idx].scatter(
            [row.axis_med], [row.e_med], color="white", ec="grey", zorder=10
        )

        ax[ax_idx].errorbar(
            [row.axis_med],
            [row.e_med],
            xerr=([row.axis_med - row.axis_minus], [row.axis_plus - row.axis_med]),
            yerr=([row.e_med - row.e_minus], [row.e_plus - row.e_med]),
            color="grey",
        )

plt.tight_layout()
plt.savefig("plots/frelikh_compare_completeness.png", dpi=250)

"""
KDE (NOT COMPLETENESS CORRECTED) PLOT
"""

# train KDE on low-mass data
dataset = np.vstack((np.log10(lowmass_smas), lowmass_eccs))

# TODO: not quite sure how to incorporate the uncertainties into the KDE training
lowmass_kernel = gaussian_kde(dataset)  # , weights=1 / np.array(lowmass_ecc_errors))


# train KDE on high-mass data
dataset = np.vstack((np.log10(highmass_smas), highmass_eccs))
highmass_kernel = gaussian_kde(dataset)  # , weights=1 / np.array(highmass_ecc_errors))


sma2plot = np.linspace(np.log10(0.02), np.log10(6), int(1e2))

ecc2plot = np.linspace(0, 1, int(1e2))

highmass_kernel_predict = np.zeros(
    (
        len(sma) - 1,
        len(ecc) - 1,
    )
)

for i, a in enumerate(sma[:-1]):
    for j, e in enumerate(ecc[:-1]):

        highmass_kernel_predict[i, j] = highmass_kernel(np.array([np.log10(a), e]))

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 3, width_ratios=(20, 20, 1))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax = [ax0, ax1, ax2]

ax[0].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[0], mass[1]
    )
)
ax[1].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[1], mass[2]
    )
)

for a in ax[:2]:
    a.set_xscale("log")
    a.set_xlim(sma[0], sma[-1])
    a.set_ylim(0, 1)
    a.set_xlabel("$a$ [au]")
    a.set_ylabel("$e$")

ax[1].pcolormesh(
    sma,
    ecc,
    highmass_kernel_predict.T,
    shading="auto",
    vmin=0,
    vmax=2.25,
)

lowmass_kernel_predict = np.zeros(
    (
        len(sma) - 1,
        len(ecc) - 1,
    )
)

for i, a in enumerate(sma[:-1]):
    for j, e in enumerate(ecc[:-1]):

        lowmass_kernel_predict[i, j] = lowmass_kernel(np.array([np.log10(a), e]))

pc = ax[0].pcolormesh(
    sma,
    ecc,
    lowmass_kernel_predict.T,
    shading="auto",
    vmin=0,
    vmax=2.25,
)

cbar = fig.colorbar(pc, cax=ax[2])
cbar.set_label("rel. prob.")

for i, row in legacy_planets.iterrows():

    # remove false positives
    if row.status not in ["A", "R", "N"]:

        if row.mass_med > mass[1]:
            ax_idx = 1
            # highmass_eccs.append(row.e_med)
            # highmass_smas.append(row.axis_med)
            # highmass_ecc_errors.append(
            #     np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            # )

        else:
            ax_idx = 0
            # lowmass_eccs.append(row.e_med)
            # lowmass_smas.append(row.axis_med)
            # lowmass_ecc_errors.append(
            #     np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            # )

        ax[ax_idx].scatter(
            [row.axis_med], [row.e_med], color="white", ec="grey", zorder=10
        )

        ax[ax_idx].errorbar(
            [row.axis_med],
            [row.e_med],
            xerr=([row.axis_med - row.axis_minus], [row.axis_plus - row.axis_med]),
            yerr=([row.e_med - row.e_minus], [row.e_plus - row.e_med]),
            color="grey",
        )

plt.tight_layout()
plt.savefig("plots/frelikh_compare_kde.png", dpi=250)


"""
KDE (COMPLETENESS CORRECTED) PLOT
"""

# train KDE on low-mass data
dataset = np.vstack((np.log10(lowmass_smas), lowmass_eccs))
lowmass_kernel = gaussian_kde(dataset)


# train KDE on high-mass data
dataset = np.vstack((np.log10(highmass_smas), highmass_eccs))
highmass_kernel = gaussian_kde(dataset)


sma2plot = np.linspace(np.log10(0.02), np.log10(6), int(1e2))

ecc2plot = np.linspace(0, 1, int(1e2))

highmass_kernel_predict = np.zeros(
    (
        len(sma) - 1,
        len(ecc) - 1,
    )
)

for i, a in enumerate(sma[:-1]):
    for j, e in enumerate(ecc[:-1]):

        highmass_kernel_predict[i, j] = highmass_kernel(np.array([np.log10(a), e]))

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, width_ratios=(1, 1))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax = [ax0, ax1]

ax[0].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[0], mass[1]
    )
)
ax[1].set_title(
    "{} M$_{{\\mathrm{{J}}}}$ < Msini < {} M$_{{\\mathrm{{J}}}}$".format(
        mass[1], mass[2]
    )
)

for a in ax[:2]:
    a.set_xscale("log")
    a.set_xlim(sma[0], sma[-1])
    a.set_ylim(0, 1)
    a.set_xlabel("$a$ [au]")
    a.set_ylabel("$e$")

ax[1].pcolormesh(
    sma,
    ecc,
    highmass_kernel_predict.T / completeness_model_himass,
    shading="auto",
)

lowmass_kernel_predict = np.zeros(
    (
        len(sma) - 1,
        len(ecc) - 1,
    )
)

for i, a in enumerate(sma[:-1]):
    for j, e in enumerate(ecc[:-1]):

        lowmass_kernel_predict[i, j] = lowmass_kernel(np.array([np.log10(a), e]))

pc = ax[0].pcolormesh(
    sma,
    ecc,
    lowmass_kernel_predict.T / completeness_model_lowmass,
    shading="auto",
)

for i, row in legacy_planets.iterrows():

    # remove false positives
    if row.status not in ["A", "R", "N"]:

        if row.mass_med > mass[1]:
            ax_idx = 1
            highmass_eccs.append(row.e_med)
            highmass_smas.append(row.axis_med)

        else:
            ax_idx = 0
            lowmass_eccs.append(row.e_med)
            lowmass_smas.append(row.axis_med)

        ax[ax_idx].scatter(
            [row.axis_med], [row.e_med], color="white", ec="grey", zorder=10
        )

        ax[ax_idx].errorbar(
            [row.axis_med],
            [row.e_med],
            xerr=([row.axis_med - row.axis_minus], [row.axis_plus - row.axis_med]),
            yerr=([row.e_med - row.e_minus], [row.e_plus - row.e_med]),
            color="grey",
        )

plt.tight_layout()
plt.savefig("plots/frelikh_compare_completeness_corrected.png", dpi=250)
