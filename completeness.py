"""
This module builds a 3D completeness map in sma-msini-ecc, based on the 
injection recovery tests performed by BJ Fulton.

These injection-recovery test outputs are in: /home/sblunt/CLSI/completeness/recoveries_all

NOTE: BJ provides the recovered parameters as well as a binary "recovered/not recovered," 
so we could think about incorporating/checking whether "recovered" means "recovered
correctly." However, visually the countours look pretty smooth, so I'm not super
worried about this.
"""

import numpy as np
import glob
import pandas as pd
from astropy import units as u, constants as cst
import matplotlib.pyplot as plt

n_ecc_bins = 25
n_sma_bins = 25
n_msini_bins = 25

ecc = np.linspace(0, 1, n_ecc_bins)
sma = np.logspace(-1, 1, n_sma_bins)
msini = np.logspace(1, 3, n_msini_bins)  # [M_earth]

d_ecc = ecc[1:] - ecc[:-1]
d_sma = sma[1:] - sma[:-1]
d_msini = msini[1:] - msini[:-1]

recoveries = np.zeros((n_ecc_bins, n_sma_bins, n_msini_bins))
injections = np.zeros((n_ecc_bins, n_sma_bins, n_msini_bins))

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files:
    df = pd.read_csv(f)

    df["ecc_completeness_bins"] = np.nan
    df["msini_completeness_bins"] = np.nan
    df["sma_completeness_bins"] = np.nan
    for i in np.arange(len(ecc) - 1):
        df["ecc_completeness_bins"][
            ((df.inj_e.values >= ecc[i]) & (df.inj_e.values < ecc[i] + d_ecc[i]))
        ] = i

    for i in np.arange(len(ecc) - 1):
        df["msini_completeness_bins"][
            (
                (df.inj_msini.values >= msini[i])
                & (df.inj_msini.values < msini[i] + d_msini[i])
            )
        ] = i
    for i in np.arange(len(ecc) - 1):
        df["sma_completeness_bins"][
            ((df.inj_au.values >= sma[i]) & (df.inj_au.values < sma[i] + d_sma[i]))
        ] = i

    recovered_planets = df[df.recovered.values]
    unrecovered_planets = df[~df.recovered.values]

    for i, row in unrecovered_planets.iterrows():
        if not np.isnan(
            row.sma_completeness_bins
            + row.msini_completeness_bins
            + row.ecc_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.msini_completeness_bins),
            ] += 1

    for i, row in recovered_planets.iterrows():
        if not np.isnan(
            row.sma_completeness_bins
            + row.msini_completeness_bins
            + row.ecc_completeness_bins
        ):
            injections[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.msini_completeness_bins),
            ] += 1
            recoveries[
                int(row.ecc_completeness_bins),
                int(row.sma_completeness_bins),
                int(row.msini_completeness_bins),
            ] += 1
completeness = recoveries / injections
completeness_sma_msini = np.nansum(completeness, axis=0) / n_ecc_bins
completeness_ecc_sma = np.nansum(completeness, axis=2) / n_msini_bins
completeness_ecc_msini = np.nansum(completeness, axis=1) / n_sma_bins
injections_ecc_sma = np.nansum(injections, axis=2)

# compare with https://www.astroexplorer.org/details/apjsabfcc1f1
plt.figure()
plt.imshow(completeness_sma_msini.T, origin="lower")
plt.xticks(
    np.arange(len(sma)),
    map(lambda x: np.format_float_positional(x, precision=2), sma),
)
plt.yticks(
    np.arange(len(msini)),
    map(
        lambda x: np.format_float_positional(x, precision=2),
        msini * cst.M_earth / cst.M_jup,
    ),
)
plt.tick_params(axis="x", rotation=90)

plt.xlabel("sma [au]")
plt.ylabel("M$\sin{i}$ [M$_{\\mathrm{{Jup}}}$]")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/sma_msini.png", dpi=250)

plt.figure()
plt.imshow(completeness_ecc_sma.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(sma)),
    map(lambda x: np.format_float_positional(x, precision=2), sma),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("sma [au]")
plt.xlabel("ecc")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/ecc_sma.png", dpi=250)

plt.figure()
plt.imshow(completeness_ecc_msini.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(msini)),
    map(
        lambda x: np.format_float_positional(x, precision=2),
        msini * cst.M_earth / cst.M_jup,
    ),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("ecc")
plt.ylabel("M$\sin{i}$ [M$_{\\mathrm{{Jup}}}$]")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("completeness")
plt.savefig("plots/ecc_msini.png", dpi=250)

plt.figure()
plt.imshow(injections_ecc_sma.T)
plt.xticks(
    np.arange(len(ecc)),
    map(lambda x: np.format_float_positional(x, precision=2), ecc),
)
plt.yticks(
    np.arange(len(sma)),
    map(lambda x: np.format_float_positional(x, precision=2), sma),
)
plt.tick_params(axis="x", rotation=90)

plt.ylabel("sma [au]")
plt.xlabel("ecc")
plt.tight_layout()
ax_cbar = plt.colorbar()
ax_cbar.set_label("injections")
plt.savefig("plots/ecc_sma_injections.png", dpi=250)
