"""
This module builds a 3D completeness map in K-per-ecc, based on the 
injection recovery tests performed by BJ Fulton.

These injection-recovery test outputs are in: /home/sblunt/CLSI/completeness/recoveries_all

NOTE: BJ provides the recovered parameters as well as a binary "recovered/not recovered," 
so we could think about incorporating/checking whether "recovered" means "recovered
correctly." However, visually the countours look pretty smooth, so I'm not super
worried about this.

TODO: plot completeness contours for Tp/w as well 
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

n_ecc_bins = 3
n_per_bins = 3
n_k_bins = 3

ecc = np.linspace(0, 1, n_ecc_bins, endpoint=False)
per = np.linspace(0, 1e5, n_per_bins, endpoint=False)
K = np.linspace(0, 1000, n_k_bins, endpoint=False)

d_ecc = ecc[1:] - ecc[:-1]
d_K = K[1:] - K[:-1]
d_per = per[1:] - per[:-1]


recoveries = np.zeros((n_ecc_bins, n_per_bins, n_k_bins))
injections = np.zeros((n_ecc_bins, n_per_bins, n_k_bins))

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files:
    df = pd.read_csv(f)

    df["ecc_completeness_bins"] = np.nan
    df["per_completeness_bins"] = np.nan
    df["K_completeness_bins"] = np.nan
    for i in np.arange(len(ecc) - 1):

        df["ecc_completeness_bins"][
            ((df.inj_e.values >= ecc[i]) & (df.inj_e.values < ecc[i] + d_ecc[i]))
        ] = i
    # account for edge of range
    df["ecc_completeness_bins"][(df.inj_e.values >= ecc[i + 1])] = i + 1

    for i in np.arange(len(ecc) - 1):
        df["per_completeness_bins"][
            (
                (df.inj_period.values >= per[i])
                & (df.inj_period.values < per[i] + d_per[i])
            )
        ] = i
    # account for edge of range
    df["per_completeness_bins"][(df.inj_period.values >= per[i + 1])] = i + 1

    for i in np.arange(len(ecc) - 1):
        df["K_completeness_bins"][
            ((df.inj_k.values >= K[i]) & (df.inj_k.values < K[i] + d_K[i]))
        ] = i
    # account for edge of range
    df["K_completeness_bins"][(df.inj_k.values >= K[i + 1])] = i + 1

    recovered_planets = df[df.recovered.values]
    unrecovered_planets = df[~df.recovered.values]

    for i, row in unrecovered_planets.iterrows():
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
completeness = recoveries / injections
print(completeness)
# completeness = np.nan_to_num(completeness)  # a few

# save completeness map for use in epop
np.save("completeness.npy", completeness)
np.save("e_bins.npy", ecc)
np.save("K_bins.npy", K)
np.save("per_bins.npy", per)

completeness_K_per = np.nansum(completeness, axis=0) / n_ecc_bins
completeness_ecc_K = np.nansum(completeness, axis=2) / n_per_bins
completeness_ecc_per = np.nansum(completeness, axis=1) / n_k_bins
injections_ecc_K = np.nansum(injections, axis=2)

# compare with https://www.astroexplorer.org/details/apjsabfcc1f1
plt.figure()
plt.imshow(completeness_K_per.T, origin="lower")
plt.xticks(
    np.arange(len(K)),
    map(lambda x: np.format_float_positional(x, precision=2), K),
)
plt.yticks(
    np.arange(len(per)),
    map(
        lambda x: np.format_float_positional(x, precision=2),
        per,  # * cst.M_earth / cst.M_jup,
    ),
)
plt.tick_params(axis="x", rotation=90)

plt.xlabel("K [m/s]")
plt.ylabel("P [d]")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/completeness_K_per.png", dpi=250)

plt.figure()
plt.imshow(completeness_ecc_K.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(K)),
    map(lambda x: np.format_float_positional(x, precision=2), K),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("K [m/s]")
plt.xlabel("ecc")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/completeness_ecc_K.png", dpi=250)

plt.figure()
plt.imshow(completeness_ecc_per.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(per)),
    map(lambda x: np.format_float_positional(x, precision=2), per),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("ecc")
plt.ylabel("P [d]")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/completeness_ecc_per.png", dpi=250)

plt.figure()
plt.imshow(injections_ecc_K.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(K)),
    map(lambda x: np.format_float_positional(x, precision=2), K),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("K [m/s]")
plt.xlabel("ecc")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("injections")
plt.savefig("plots/ecc_K_injections.png", dpi=250)
