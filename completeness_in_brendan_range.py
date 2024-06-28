"""
This module builds a 1D completeness map as a function of eccentricity for RV planets
in the mass range 2-15 Mjup and semimajor axis range 5-100 au. 

Based on the injection recovery tests performed by BJ Fulton.
"""

import numpy as np
import glob
import pandas as pd
from astropy import units as u, constants as cst
import matplotlib.pyplot as plt

n_ecc_bins = 20

ecc = np.linspace(0, 1, n_ecc_bins)

d_ecc = ecc[1:] - ecc[:-1]

recoveries = np.zeros(n_ecc_bins)
injections = np.zeros(n_ecc_bins)

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

    recovered_planets = df[df.recovered.values]
    unrecovered_planets = df[~df.recovered.values]

    for i, row in unrecovered_planets.iterrows():
        # check if injected planet is in Brendan's sample range
        if (row.inj_au >= 5) & (row.inj_au < 100):
            if (row.inj_msini >= (2 * u.M_jup / u.M_earth).to("").value) & (
                row.inj_msini < (15 * u.M_jup / u.M_earth).to("").value
            ):
                injections[int(row.ecc_completeness_bins)] += 1

    for i, row in recovered_planets.iterrows():
        # check if injected planet is in Brendan's sample range
        if (row.inj_au >= 5) & (row.inj_au < 100):
            if (row.inj_msini >= (2 * u.M_jup / u.M_earth).to("").value) & (
                row.inj_msini < (15 * u.M_jup / u.M_earth).to("").value
            ):
                injections[int(row.ecc_completeness_bins)] += 1
                recoveries[int(row.ecc_completeness_bins)] += 1

# compute uncertainty on completeness (using Poisson statistics from # of injections)
completeness = recoveries / injections
uncertainty = 1 / np.sqrt(injections)

plt.figure()
plt.plot(ecc, completeness, color="rebeccapurple")
plt.fill_between(
    ecc,
    completeness + uncertainty,
    completeness - uncertainty,
    color="rebeccapurple",
    alpha=0.2,
)

# overplot a linear fit
(
    m,
    b,
) = np.polyfit(ecc, completeness, 1, w=1 / uncertainty)
plt.plot(ecc, m * ecc + b, color="k", ls="--", label="m={:.3f}, b={:.3f}".format(m, b))
plt.legend()
plt.title("RV planets in same range as Brendan planets")
plt.xlabel("eccentricity")
plt.ylabel("RV completeness")
plt.savefig("plots/completeness_ecc.png", dpi=250)
