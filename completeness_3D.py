"""
This module builds a 1D completeness map as a function of eccentricity for RV planets
in the mass range 2-15 Mjup and semimajor axis range 5-100 au. 

Based on the injection recovery tests performed by BJ Fulton.
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


n_ecc_bins = 19
n_per_bins = 21
n_k_bins = 23


ecc = np.linspace(0, 1, n_ecc_bins + 1)
per = np.logspace(np.log10(300), np.log10(1e6), n_per_bins + 1)
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

# compute marginalized completenesses
completeness_K_per = np.nansum(completeness, axis=0) / n_ecc_bins
completeness_per = np.nansum(completeness_K_per, axis=0) / n_k_bins
completeness_ecc_K = np.nansum(completeness, axis=2) / n_per_bins
completeness_ecc_per = np.nansum(completeness, axis=1) / n_k_bins

# perform linear fit
n_features = 3
X = np.ones((n_ecc_bins, n_k_bins, n_per_bins, n_features))
for i in np.arange(n_ecc_bins):
    for j in np.arange(n_k_bins):
        for k in np.arange(n_per_bins):
            X[i, j, k, 0] = ecc[i]
            X[i, j, k, 1] = np.log(
                K[j]
            )  # TODO: I want 1/per, uniform in log(K), possibly (log(K))**2
            X[i, j, k, 2] = 1 / per[k]

X = X.reshape((n_ecc_bins * n_k_bins * n_per_bins, n_features))
completeness1d = completeness.flatten()
log_completeness = np.log(completeness1d)

# weight samples using Poisson statistics from # of injections.
# this weight should be 1 / the variance of each number. assuming
# Poisson statistics, the variance in N, the number of samples.
sample_weights = 1 / injections

# only include boxes with 0 injections or 0 recoveries in the linear fit
mask = ((injections > 0) & (completeness > 0)).flatten()

reg = LinearRegression().fit(
    X[mask],
    log_completeness[mask],
    sample_weight=sample_weights.flatten()[mask],
)

"""
K-PER PLOT
"""
model_completeness_K_per = np.zeros((n_k_bins, n_per_bins))
for i in np.arange(n_k_bins):
    for j in np.arange(n_per_bins):
        X_pred = np.array(
            [
                0.5,
                np.log(K[i]),
                1 / per[j],
            ]
        )
        model_completeness_K_per[i, j] = np.exp(
            reg.predict(X_pred.reshape((1, n_features)))
        )

K_per_mask = (completeness_K_per == 0) | (np.isnan(completeness_K_per))

fig, ax = plt.subplots(2, 1, figsize=(5, 10))

ax[0].pcolormesh(K, per, model_completeness_K_per.T, shading="auto")
ax[1].pcolormesh(
    K, per, np.ma.array(completeness_K_per.T, mask=K_per_mask.T), shading="auto"
)

ax[0].set_title("completeness model")
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
