"""
Make a model of survey completeness by averaging all injection-recovery tests
across stars in sample and interpolating missing values (if any)
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import copy
import astropy.units as u, astropy.constants as cst
import os

n_ecc_bins = 5
n_sma_bins = 1
n_msini_bins = 3  # [<1.17 Mj and > 1.17Mj is binning used in Frelikh+]
path_to_clsi = f"{os.path.expanduser('~')}/Documents/Github"  # where I stored the clsi Github repo with inj-rec tests

# NOTE: BIN BOUNDARIES DEFINED HERE
ecc_bin_edges = np.linspace(0, 1, n_ecc_bins + 1)

# these sma/msini limits overlap with BJ's bins
sma_bin_edges = np.logspace(np.log10(0.10533575), np.log10(4.55973325), n_sma_bins + 1)

"""
# this is what BJ uses (plus the bin I added for the brown dwarfs). I don't see a peak with these bins,
# (which leads me to believe the peak occurs at higher masses)

msini_bin_edges = np.array(
    [30, 300, 6_000, 30_000]
)  
"""

msini_bin_edges = np.array([30, 1_000, 6_000, 30_000])

recoveries = np.zeros((n_ecc_bins, n_sma_bins, n_msini_bins))
injections = np.zeros((n_ecc_bins, n_sma_bins, n_msini_bins))

inj_rec_files = glob.glob(f"{path_to_clsi}/CLSI/completeness/recoveries_all/*.csv")
for k, f in enumerate(inj_rec_files):
    print(f"Reading {k}/{len(inj_rec_files)}", end="\r")
    df = pd.read_csv(f)

    df["ecc_completeness_bins"] = np.nan
    df["sma_completeness_bins"] = np.nan
    df["mass_completeness_bins"] = np.nan
    for i in np.arange(n_ecc_bins):
        df.loc[
            (
                (df.inj_e.values >= ecc_bin_edges[i])
                & (df.inj_e.values < ecc_bin_edges[i + 1])
            ),
            "ecc_completeness_bins",
        ] = i
    for i in np.arange(n_sma_bins):

        df.loc[
            (
                (df.inj_au.values >= sma_bin_edges[i])
                & (df.inj_au.values < sma_bin_edges[i + 1])
            ),
            "sma_completeness_bins",
        ] = i

    for i in np.arange(n_msini_bins):
        df.loc[
            (
                (df.inj_msini.values >= msini_bin_edges[i])
                & (df.inj_msini.values < msini_bin_edges[i + 1])
            ),
            "mass_completeness_bins",
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
print()
print(f"Injections in each bin: {injections[:, :, :-1]}")
print(f"Recoveries in each bin: {recoveries[:, :, :-1]}")

# values in the 3d grid where there are 0 injections or 0 recoveries
bad_mask = (completeness == 0) | (np.isnan(completeness))

print(
    "Interpolated {} values ({:.2f}%)".format(
        np.sum(bad_mask), np.sum(bad_mask) / len(completeness.flatten()) * 100
    )
)

bad_idx = np.where(bad_mask)

# get (e,sma,msini) coords where we need to interpolate values
iterp_here = []

for idx, i in enumerate(bad_idx[0]):

    iterp_here.append([i, bad_idx[1][idx], bad_idx[2][idx]])

filled_in_points = scipy.interpolate.interpn(
    (ecc_bin_edges[:-1], sma_bin_edges[:-1], msini_bin_edges[:-1]),
    np.ma.array(completeness, mask=bad_mask),
    np.array(iterp_here),
    bounds_error=False,
    method="linear",
)

completeness_model = copy.copy(completeness)
completeness_model[bad_mask] = filled_in_points

# save the completeness model
np.save(
    "completeness_model/{}{}{}completeness".format(
        n_msini_bins, n_ecc_bins, n_sma_bins
    ),
    completeness_model,
)
np.save(
    "completeness_model/{}msini_bins".format(n_msini_bins),
    msini_bin_edges,
)
np.save("completeness_model/{}ecc_bins".format(n_ecc_bins), ecc_bin_edges)
np.save("completeness_model/{}sma_bins".format(n_sma_bins), sma_bin_edges)

# completeness_model_medmass = completeness_model[:, :, -2]
# completeness_model_himass = completeness_model[:, :, -1]


"""
COMPLETENESS PLOT 
"""

fig = plt.figure(figsize=(5 * n_msini_bins, 5))
gs = fig.add_gridspec(1, n_msini_bins + 1, width_ratios=[20] * n_msini_bins + [1])
ax = []
for i in range(n_msini_bins + 1):
    ax.append(fig.add_subplot(gs[0, i]))

for i, a in enumerate(ax[:-1]):
    a.set_title(
        "{:.1f} M$_{{\\oplus}}$ < Msini < {:.1f} M$_{{\\oplus}}$".format(
            msini_bin_edges[i], msini_bin_edges[i + 1]
        )
    )

for i in np.arange(n_msini_bins):
    pc = ax[i].pcolormesh(
        sma_bin_edges,
        ecc_bin_edges,
        completeness_model[:, :, i],
        shading="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
        alpha=0.5,
        edgecolor="k",
    )
cbar = fig.colorbar(pc, cax=ax[-1])
cbar.set_label("completeness")

for a in ax[:-1]:
    a.set_xscale("log")
    a.set_xlim(sma_bin_edges[0], sma_bin_edges[-1])
    a.set_ylim(0, 1)
    a.set_xlabel("$a$ [au]")
ax[0].set_ylabel("$e$")


# overplot importance sampled planet params from CLS
origin = "resampled"
for post_path in glob.glob(f"lee_posteriors/{origin}/ecc_*.csv"):

    ecc_post = pd.read_csv(post_path).values.flatten()
    post_len = len(ecc_post)

    st_name = post_path.split("/")[-1].split("_")[1]
    pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0]

    msini_post = pd.read_csv(
        f"lee_posteriors/{origin}/msini_{st_name}_{pl_num}.csv"
    ).values.flatten()
    sma_post = pd.read_csv(
        f"lee_posteriors/{origin}/sma_{st_name}_{pl_num}.csv"
    ).values.flatten()

    ax_idx = None
    for i in np.arange(n_msini_bins):
        if (
            np.median(msini_post) > msini_bin_edges[i]
            and np.median(msini_post) < msini_bin_edges[i + 1]
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
            color="k",
            alpha=0.5,
            lw=2,
        )

        for j, a in enumerate(sma_bin_edges[:-1]):
            for k, e in enumerate(ecc_bin_edges[:-1]):

                ax[ax_idx].text(
                    a,
                    e + 0.02,
                    "{:.2f} ".format(completeness_model[k, j, ax_idx]),
                    color="k",
                    zorder=20,
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
                )

"""
# overplot the published limits from Lee's paper (if you want to see how little the importance resampling matters,
# and get really dizzy in the process)

lowmass_eccs = []
lowmass_smas = []
highmass_eccs = []
highmass_smas = []
lowmass_ecc_errors = []
highmass_ecc_errors = []
legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0, comment="#"
)
for i, row in legacy_planets.iterrows():

    # remove false positives
    if row.status not in ["A", "R", "N"]:

        if (row.mass_med * u.M_jup / u.M_earth).to("") > mass[-2] and (
            row.mass_med * u.M_jup / u.M_earth
        ).to("") < mass[-1]:
            ax_idx = 1
            highmass_eccs.append(row.e_med)
            highmass_smas.append(row.axis_med)
            highmass_ecc_errors.append(
                np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            )

        elif (row.mass_med * u.M_jup / u.M_earth).to("") < mass[-2] and (
            row.mass_med * u.M_jup / u.M_earth
        ).to("") > mass[-3]:
            ax_idx = 0
            lowmass_eccs.append(row.e_med)
            lowmass_smas.append(row.axis_med)
            lowmass_ecc_errors.append(
                np.max([row.e_plus - row.e_med, row.e_med - row.e_minus])
            )

ax[ax_idx].errorbar(
    [row.axis_med],
    [row.e_med],
    xerr=([row.axis_med - row.axis_minus], [row.axis_plus - row.axis_med]),
    yerr=([row.e_med - row.e_minus], [row.e_plus - row.e_med]),
    color="k",
    lw=0.5,
    alpha=1,
)
"""

plt.tight_layout()

fullmarg = "fullmarg_"  # ['fullmarg_', '']

savedir = f"plots/{fullmarg}{n_msini_bins}msini{n_sma_bins}sma{n_ecc_bins}e"
print(savedir)
if not os.path.exists(savedir):
    os.mkdir(savedir)

plt.savefig(f"{savedir}/completeness.png", dpi=250)
