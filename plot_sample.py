import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Chill plot to explore the sample
"""

mass_bin_edges = np.array([30, 600, 6_000, 300_000])
# 600 Mearth = 1.9 Mjup, 6_000 Mearth= 19 Mjup, 300_000 Mearth = 0.9 Msun
n_mass_bins = len(mass_bin_edges) - 1

sma_bound = [0.1, 5]

fig, ax = plt.subplots(1, n_mass_bins, figsize=(5 * n_mass_bins, 5))

total_detections = 0
for post_path in glob.glob("lee_posteriors/resampled/ecc_*.csv"):

    ecc_post = pd.read_csv(post_path).values.flatten()
    post_len = len(ecc_post)

    st_name = post_path.split("/")[-1].split("_")[1]
    pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0]

    msini_post = pd.read_csv(
        f"lee_posteriors/resampled/msini_{st_name}_{pl_num}.csv"
    ).values.flatten()
    sma_post = pd.read_csv(
        f"lee_posteriors/resampled/sma_{st_name}_{pl_num}.csv"
    ).values.flatten()

    ax_idx = None
    for i in np.arange(n_mass_bins):
        if (
            mass_bin_edges[i] < np.median(msini_post)
            and np.median(msini_post) < mass_bin_edges[i + 1]
            and np.median(sma_post) < sma_bound[-1]
            and np.median(sma_post) > sma_bound[0]
        ):
            ax_idx = i
            total_detections += 1

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

for a in ax:
    a.set_xscale("log")

for i, a in enumerate(ax):
    a.set_title(
        "{:.1f} M$_{{\\oplus}}$ < Msini < {:.1f} M$_{{\\oplus}}$".format(
            mass_bin_edges[i], mass_bin_edges[i + 1]
        )
    )
ax[0].text(1e-1, 0.8, f"total detections: {total_detections}")
plt.tight_layout()
plt.savefig("plots/sanity_checks/sample_split.png", dpi=250)
