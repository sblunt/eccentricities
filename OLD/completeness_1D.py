"""
This module builds a 1D completeness map as a function of eccentricity for RV planets
in the mass range 2-15 Mjup and semimajor axis range 5-100 au. 

Based on the injection recovery tests performed by BJ Fulton.
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

n_ecc_bins = 25
n_per_bins = 25
n_k_bins = 25


ecc = np.linspace(0, 1, n_ecc_bins + 1)
per = np.linspace(300, 1e6, n_per_bins + 1)
K = np.logspace(
    1,
    np.log10(1000),
    n_k_bins + 1,
)

per_min = np.min(per)
per_max = np.max(per)
K_min = np.min(K)
K_max = np.max(K)

recoveries = np.zeros(n_ecc_bins)
injections = np.zeros(n_ecc_bins)

recoveries_per = np.zeros(n_per_bins)
injections_per = np.zeros(n_per_bins)

recoveries_k = np.zeros(n_k_bins)
injections_k = np.zeros(n_k_bins)

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files[:100]:
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
        if (row.inj_k >= K_min) & (row.inj_k < K_max):
            if (row.inj_period >= per_min) & (row.inj_period < per_max):
                injections[int(row.ecc_completeness_bins)] += 1
                injections_k[int(row.K_completeness_bins)] += 1
                injections_per[int(row.per_completeness_bins)] += 1

    for i, row in recovered_planets.iterrows():
        # check if injected planet is in range
        if (row.inj_k >= K_min) & (row.inj_k < K_max):
            if (row.inj_period >= per_min) & (row.inj_period < per_max):
                injections[int(row.ecc_completeness_bins)] += 1
                recoveries[int(row.ecc_completeness_bins)] += 1

                injections_k[int(row.K_completeness_bins)] += 1
                recoveries_k[int(row.K_completeness_bins)] += 1

                injections_per[int(row.per_completeness_bins)] += 1
                recoveries_per[int(row.per_completeness_bins)] += 1

# compute uncertainty on completeness (using Poisson statistics from # of injections)
completeness = recoveries / injections
uncertainty = 1 / np.sqrt(injections)

plt.figure()
plt.plot(ecc[:-1], completeness, color="rebeccapurple")
plt.fill_between(
    ecc[:-1],
    completeness + uncertainty,
    completeness - uncertainty,
    color="rebeccapurple",
    alpha=0.2,
)

# overplot a linear fit
(
    m,
    b,
) = np.polyfit(ecc[:-1], completeness, 1, w=1 / uncertainty)
plt.plot(
    ecc[:-1],
    m * ecc[:-1] + b,
    color="k",
    ls="--",
    label="m={:.3f}, b={:.3f}".format(m, b),
)
plt.legend()
plt.xlabel("eccentricity")
plt.ylabel("RV completeness")
plt.savefig("plots/completeness_ecc.png", dpi=250)


completeness = recoveries_per / injections_per
uncertainty = 1 / np.sqrt(injections_per)

plt.figure()
plt.plot(per[:-1], completeness, color="rebeccapurple")
plt.fill_between(
    per[:-1],
    completeness + uncertainty,
    completeness - uncertainty,
    color="rebeccapurple",
    alpha=0.2,
)
# plt.xscale("log")
plt.xlabel("period [d]")
plt.ylabel("RV completeness")
plt.savefig("plots/completeness_per.png", dpi=250)

completeness = recoveries_k / injections_k
uncertainty = 1 / np.sqrt(injections_k)

plt.figure()
plt.plot(K[:-1], completeness, color="rebeccapurple")
plt.fill_between(
    K[:-1],
    completeness + uncertainty,
    completeness - uncertainty,
    color="rebeccapurple",
    alpha=0.2,
)
plt.xscale("log")
plt.xlabel("K [m/s]")
plt.ylabel("RV completeness")
plt.savefig("plots/completeness_K.png", dpi=250)
