"""
This module builds a 1D completeness map as a function of eccentricity for RV planets
in the mass range 2-15 Mjup and semimajor axis range 5-100 au. 

Based on the injection recovery tests performed by BJ Fulton.
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import copy

n_ecc_bins = 15
n_per_bins = 15
n_k_bins = 15


ecc = np.linspace(0, 1, n_ecc_bins + 1)
per = np.logspace(np.log10(300), 6, n_per_bins + 1)
K = np.logspace(
    1,
    np.log10(1000),
    n_k_bins + 1,
)

recoveries = np.zeros((n_ecc_bins, n_k_bins, n_per_bins))
injections = np.zeros((n_ecc_bins, n_k_bins, n_per_bins))

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files:
    df = pd.read_csv(f)

    df["ecc_completeness_bins"] = np.nan
    df["per_completeness_bins"] = np.nan
    df["K_completeness_bins"] = np.nan
    for i in np.arange(len(ecc) - 1):
        df["ecc_completeness_bins"][
            ((df.inj_e.values >= ecc[i]) & (df.inj_e.values < ecc[i + 1]))
        ] = i
    for i in np.arange(len(per) - 1):
        df["per_completeness_bins"][
            ((df.inj_period.values >= per[i]) & (df.inj_period.values < per[i + 1]))
        ] = i
    for i in np.arange(len(per) - 1):
        df["K_completeness_bins"][
            ((df.inj_k.values >= K[i]) & (df.inj_k.values < K[i + 1]))
        ] = i

    recovered_planets = df[df.recovered.values]
    unrecovered_planets = df[~df.recovered.values]

    for i, row in unrecovered_planets.iterrows():
        # check if injected planet is in range
        if not np.isnan(
            row.K_completeness_bins
            + row.per_completeness_bins
            + row.ecc_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.K_completeness_bins),
                int(row.per_completeness_bins),
            ] += 1

    for i, row in recovered_planets.iterrows():
        # check if injected planet is in range
        if not np.isnan(
            row.K_completeness_bins
            + row.per_completeness_bins
            + row.ecc_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.K_completeness_bins),
                int(row.per_completeness_bins),
            ] += 1
            recoveries[
                int(row.ecc_completeness_bins),
                int(row.K_completeness_bins),
                int(row.per_completeness_bins),
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

# get (e,K,per) coords where we need to interpolate values
iterp_here = []
for i in bad_idx[0]:
    iterp_here.append([i, bad_idx[1][i], bad_idx[2][i]])

filled_in_points = scipy.interpolate.interpn(
    (ecc[:-1], K[:-1], per[:-1]),
    np.ma.array(completeness, mask=bad_mask),
    np.array(iterp_here),
    bounds_error=False,
    fill_value=0,
)

completeness_model = copy.copy(completeness)
completeness_model[bad_mask] = filled_in_points

# save the completeness model
np.save("completeness_model/completeness.npz", completeness_model)
np.save("completeness_model/ecc_bins.npz", ecc)
np.save("completeness_model/K_bins.npz", K)
np.save("completeness_model/per_bins.npz", per)

# compute marginalized completenesses
completeness_K_per = np.nansum(completeness, axis=0) / n_ecc_bins
completeness_model_K_per = np.nansum(completeness_model, axis=0) / n_ecc_bins
completeness_ecc_per = np.nansum(completeness, axis=1) / n_k_bins
completeness_model_ecc_per = np.nansum(completeness_model, axis=1) / n_per_bins
completeness_ecc_K = np.nansum(completeness, axis=2) / n_per_bins
completeness_model_ecc_K = np.nansum(completeness_model, axis=2) / n_per_bins

"""
K-PER PLOT
"""

K_per_mask = (completeness_K_per == 0) | (np.isnan(completeness_K_per))

fig, ax = plt.subplots(2, 1, figsize=(5, 10))

ax[0].pcolormesh(K, per, completeness_model_K_per.T, shading="auto", vmin=0, vmax=1)
ax[1].pcolormesh(
    K,
    per,
    np.ma.array(completeness_K_per.T, mask=K_per_mask.T),
    shading="auto",
    vmin=0,
    vmax=1,
)

ax[0].set_title("interpolated completeness")
ax[1].set_title("actual injections/recoveries")

for a in ax:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("K [m/s]")
    a.set_ylabel("per [d]")
    a.set_xlim(K[0], K[-1])
    a.set_ylim(per[0], per[-1])

# overplot actual detections
for post_path in glob.glob("lee_posteriors/*/ecc_*.csv"):
    pl_name = post_path.split("/")[-1].split(".")[0].split("ecc_")[1]
    category = post_path.split("/")[-2]
    per_post = pd.read_csv(f"lee_posteriors/{category}/per_{pl_name}.csv")
    K_post = pd.read_csv(f"lee_posteriors/{category}/K_{pl_name}.csv")

    per_quants = np.quantile(per_post, [0.16, 0.5, 0.84])
    K_quants = np.quantile(K_post, [0.16, 0.5, 0.84])

    plt.scatter([K_quants[1]], [per_quants[1]], color="white", ec="grey", zorder=10)
    plt.errorbar(
        [K_quants[1]],
        [per_quants[1]],
        xerr=([K_quants[1] - K_quants[0]], [K_quants[2] - K_quants[1]]),
        yerr=([per_quants[1] - per_quants[0]], [per_quants[2] - per_quants[1]]),
        color="grey",
    )

plt.savefig("plots/completeness_K_per.png", dpi=250)

"""
K-ECC PLOT
"""

ecc_K_mask = (completeness_ecc_K == 0) | (np.isnan(completeness_ecc_K))

fig, ax = plt.subplots(2, 1, figsize=(5, 10))

ax[0].pcolormesh(ecc, K, completeness_model_ecc_K.T, shading="auto", vmin=0, vmax=1)
ax[1].pcolormesh(
    ecc,
    K,
    np.ma.array(completeness_ecc_K.T, mask=ecc_K_mask.T),
    shading="auto",
    vmin=0,
    vmax=1,
)

ax[0].set_title("interpolated completeness")
ax[1].set_title("actual injections/recoveries")

for a in ax:
    a.set_yscale("log")
    a.set_xlabel("ecc")
    a.set_ylabel("K [m/s]")
    a.set_xlim(0, 1)
    a.set_ylim(K[0], K[-1])

# overplot actual detections
for post_path in glob.glob("lee_posteriors/*/ecc_*.csv"):
    pl_name = post_path.split("/")[-1].split(".")[0].split("ecc_")[1]
    category = post_path.split("/")[-2]
    ecc_post = pd.read_csv(f"lee_posteriors/{category}/ecc_{pl_name}.csv")
    K_post = pd.read_csv(f"lee_posteriors/{category}/K_{pl_name}.csv")

    ecc_quants = np.quantile(ecc_post, [0.16, 0.5, 0.84])
    K_quants = np.quantile(K_post, [0.16, 0.5, 0.84])

    plt.scatter([ecc_quants[1]], [K_quants[1]], color="white", ec="grey", zorder=10)
    plt.errorbar(
        [ecc_quants[1]],
        [K_quants[1]],
        xerr=([ecc_quants[1] - ecc_quants[0]], [ecc_quants[2] - ecc_quants[1]]),
        yerr=([K_quants[1] - K_quants[0]], [K_quants[2] - K_quants[1]]),
        color="grey",
    )

plt.savefig("plots/completeness_ecc_K.png", dpi=250)


"""
ECC-PER PLOT
"""

ecc_per_mask = (completeness_ecc_per == 0) | (np.isnan(completeness_ecc_per))

fig, ax = plt.subplots(2, 1, figsize=(5, 10))

ax[0].pcolormesh(ecc, per, completeness_model_ecc_per.T, shading="auto", vmin=0, vmax=1)
ax[1].pcolormesh(
    ecc,
    per,
    np.ma.array(completeness_ecc_per.T, mask=ecc_per_mask.T),
    shading="auto",
    vmin=0,
    vmax=1,
)

ax[0].set_title("interpolated completeness")
ax[1].set_title("actual injections/recoveries")

for a in ax:
    a.set_yscale("log")
    a.set_xlabel("ecc")
    a.set_ylabel("P [d]")
    a.set_xlim(0, 1)
    a.set_ylim(per[0], per[-1])

# overplot actual detections
for post_path in glob.glob("lee_posteriors/*/ecc_*.csv"):
    pl_name = post_path.split("/")[-1].split(".")[0].split("ecc_")[1]
    category = post_path.split("/")[-2]
    ecc_post = pd.read_csv(f"lee_posteriors/{category}/ecc_{pl_name}.csv")
    per_post = pd.read_csv(f"lee_posteriors/{category}/per_{pl_name}.csv")

    ecc_quants = np.quantile(ecc_post, [0.16, 0.5, 0.84])
    per_quants = np.quantile(per_post, [0.16, 0.5, 0.84])

    plt.scatter([ecc_quants[1]], [per_quants[1]], color="white", ec="grey", zorder=10)
    plt.errorbar(
        [ecc_quants[1]],
        [per_quants[1]],
        xerr=([ecc_quants[1] - ecc_quants[0]], [ecc_quants[2] - ecc_quants[1]]),
        yerr=([per_quants[1] - per_quants[0]], [per_quants[2] - per_quants[1]]),
        color="grey",
    )

plt.savefig("plots/completeness_ecc_per.png", dpi=250)
