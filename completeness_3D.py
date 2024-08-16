"""
This module builds a 3D completeness map in K-per-ecc, based on the 
injection recovery tests performed by BJ Fulton.

These injection-recovery test outputs are in: /home/sblunt/CLSI/completeness/recoveries_all

NOTE: BJ provides the recovered parameters as well as a binary "recovered/not recovered," 
so we could think about incorporating/checking whether "recovered" means "recovered
correctly." However, visually the countours look pretty smooth, so I'm not super
worried about this.

K values of pls in sample: 37, 46, 121, 170, 14, 52, 196, 295, 25, 26, 130, 35, 28, 85
-> range = 10-300

P values of pls in sample: 4e4, 2.5e4, 4.7e3, 7e3, 9.5e4, 5e3, 2e4, 2e4, 7e3, 1e4, 3e4, 4e3, 8e3, 5e3
-> range = 1e3 - 2e5
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

n_ecc_bins = 15
n_per_bins = 15
n_k_bins = 15

ecc = np.linspace(0, 1, n_ecc_bins, endpoint=False)
per = np.linspace(1e3, 2e5, n_per_bins, endpoint=False)
K = np.linspace(10, 300, n_k_bins, endpoint=False)

d_ecc = ecc[1:] - ecc[:-1]
d_K = K[1:] - K[:-1]
d_per = per[1:] - per[:-1]


recoveries = np.zeros((n_ecc_bins, n_per_bins, n_k_bins))
injections = np.zeros((n_ecc_bins, n_per_bins, n_k_bins))

inj_rec_files = glob.glob("/home/sblunt/CLSI/completeness/recoveries_all/*.csv")
for f in inj_rec_files[0:50]:
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

    for i in np.arange(len(per) - 1):
        df["per_completeness_bins"][
            (
                (df.inj_period.values >= per[i])
                & (df.inj_period.values < per[i] + d_per[i])
            )
        ] = i
    # account for edge of range
    df["per_completeness_bins"][(df.inj_period.values >= per[i + 1])] = i + 1

    for i in np.arange(len(K) - 1):
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

X = np.ones((n_ecc_bins, n_k_bins, n_per_bins, 3))
for i in np.arange(n_ecc_bins):
    for j in np.arange(n_k_bins):
        for k in np.arange(n_per_bins):
            X[i, j, k, 0] = ecc[i]
            X[i, j, k, 1] = K[j]
            X[i, j, k, 2] = per[k]


X = X.reshape((n_ecc_bins * n_k_bins * n_per_bins, 3))

completeness1d = completeness.flatten()
mask = ~np.isnan(completeness1d)

# linear fit to completeness
reg = LinearRegression().fit(X[mask], completeness1d[mask])

print(reg.coef_)  # [-3.08887057e-01  6.32653141e-04 -2.27078094e-06]
print(reg.intercept_)  # 0.3971037479501889

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

"""
K-PER PLOT
"""

# TODO: make K-per plot (below)
# TODO: experiment with fitting completeness plane in log(per) rather than in per

# fig, ax = plt.subplots(2, 1)
# ax[0].imshow(completeness_K_per.T, origin="lower")
# ax[0].set_xticks(
#     np.arange(len(K)),
#     map(lambda x: np.format_float_positional(x, precision=2), K),
# )
# ax[0].set_yticks(
#     np.arange(len(per)),
#     map(
#         lambda x: np.format_float_positional(x, precision=2),
#         per,
#     ),
# )
# ax[0].tick_params(axis="x", rotation=90)

# ax[0].set_xlabel("K [m/s]")
# ax[0].set_ylabel("P [d]")
# plt.tight_layout()
# ax_cbar = plt.colorbar()
# ax_cbar.set_label("completeness")
# plt.savefig("plots/completeness_K_per.png", dpi=250)

"""
ECC-K PLOT
"""

model_completeness_ecc_K = np.zeros((n_ecc_bins, n_k_bins))
for i in np.arange(n_ecc_bins):
    for j in np.arange(n_k_bins):
        X_pred = np.array([ecc[i], K[j], 0.5 * (per[-1] - per[0])])
        model_completeness_ecc_K[i, j] = reg.predict(X_pred.reshape((1, 3)))

fig, ax = plt.subplots(2, 1)
ax[0].imshow(model_completeness_ecc_K.T)
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

"""
ECC-PER PLOT
"""
model_completeness_ecc_per = np.zeros((n_ecc_bins, n_per_bins))
for i in np.arange(n_ecc_bins):
    for j in np.arange(n_per_bins):
        X_pred = np.array([ecc[i], 0.5 * (K[-1] - K[0]), per[j]])
        model_completeness_ecc_per[i, j] = reg.predict(X_pred.reshape((1, 3)))

fig, ax = plt.subplots(2, 1)
ax[0].imshow(model_completeness_ecc_per.T)
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
