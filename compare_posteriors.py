import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

"""
Compare low vs high-mass outputs for a given prior/completeness model
"""

hprior = "None"
oneDcompleteness = False

savedir = f"plots/{hprior}Prior"

lowmass_savedir = savedir + f"_lowmass"
highmass_savedir = savedir + f"_highmass"

if oneDcompleteness:
    lowmass_savedir += "_1Dcompleteness"
    highmass_savedir += "_1Dcompleteness"

beta_post_highmass = np.loadtxt(f"{highmass_savedir}/epop_samples.csv", delimiter=",")
beta_post_lowmass = np.loadtxt(f"{lowmass_savedir}/epop_samples.csv", delimiter=",")

# pick some samples from the posterior to plot
n2plot = 50
colors = ["gold", "blue"]
median_colors = ["k", "white"]
plt.figure()
for i, beta_post in enumerate([beta_post_highmass, beta_post_lowmass]):
    idx_to_plot = np.random.choice(len(beta_post), size=n2plot)

    x2plot = np.linspace(0, 1, 200)

    for idx in idx_to_plot:
        a = beta_post[idx, 0]
        b = beta_post[idx, 1]
        plt.plot(x2plot, beta.pdf(x2plot, a, b), color=colors[i], alpha=0.2)
    plt.plot(
        x2plot,
        beta.pdf(x2plot, np.median(beta_post[:, 0]), np.median(beta_post[:, 1])),
        color=median_colors[i],
        lw=1,
        ls="--",
    )

plt.xlim(0, 1)
plt.xlabel("eccentricity")
plt.ylabel("prob. density")
plt.savefig(
    "plots/lowhighcompare_{}_1dcompleteness{}.png".format(hprior, oneDcompleteness),
    dpi=250,
)
