""""
Plot results from an epop run
"""

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import corner

# read in MCMC samples
hprior = "gaussian"
oneDcompleteness = False
samples = [
    # "close_bds",
    # "far_bds",
    # "close_planets",
    "far_planets"
]

savedir = f"plots/{hprior}Prior"

for sam in samples:
    savedir += f"_{sam}"

if oneDcompleteness:
    savedir += "_1Dcompleteness"

beta_post = np.loadtxt(f"{savedir}/epop_samples.csv", delimiter=",")

"""
trend plot
"""
fig, ax = plt.subplots(2, 1, figsize=(25, 5), sharex=True)
plt.subplots_adjust(hspace=0)
nwalkers = 50
chains = beta_post.reshape((-1, nwalkers, 2))

for i in np.arange(nwalkers):
    ax[0].plot(chains[:, i, 0], alpha=0.05, color="k")
    ax[1].plot(chains[:, i, 1], alpha=0.05, color="k")
plt.savefig(f"{savedir}/trend.png", dpi=250)

"""
Eccentricity samples plot
"""

# pick some samples from the posterior to plot
n2plot = 200
idx_to_plot = np.random.choice(len(beta_post), size=n2plot)

plt.figure()
x2plot = np.linspace(0, 1, 150)

# TODO: this is a bit of a hack-- I should really get the median and 1sigma a/b of the whole posterior,
# not the 1d marginalizerd ones

# for idx in idx_to_plot:
#     a = beta_post[idx, 0]
#     b = beta_post[idx, 1]
#     plt.plot(x2plot, beta.pdf(x2plot, a, b), color="gold", alpha=0.2)
# plt.plot(
#     x2plot,
#     beta.pdf(x2plot, np.median(beta_post[:, 0]), np.median(beta_post[:, 1])),
#     color="k",
#     lw=2,
#     label="RV planets",
# )
# a_quants = np.quantile(beta_post[:, 0], [0.16, 0.5, 0.84])
# b_quants = np.quantile(beta_post[:, 1], [0.16, 0.5, 0.84])
# plt.plot(x2plot, beta.pdf(x2plot, a_quants[0], b_quants[0]))
# plt.plot(x2plot, beta.pdf(x2plot, a_quants[2], b_quants[0]))
# plt.plot(
#     x2plot, beta.pdf(x2plot, a_quants[0], b_quants[0]), ls="--", color="k", alpha=0.2
# )
# plt.plot(
#     x2plot, beta.pdf(x2plot, a_quants[2], b_quants[2]), ls="--", color="k", alpha=0.2
# )
# plt.fill_between(
#     x2plot,
#     beta.pdf(x2plot, a_quants[2], b_quants[2]),
#     beta.pdf(x2plot, a_quants[0], b_quants[0]),
#     alpha=0.2,
#     color="k",
# )

# overplot the imaged planet dist
# a = np.random.normal(0.7, 0.3, size=n2plot)
# b = np.random.normal(2.3, 0.7, size=n2plot)
plt.plot(
    x2plot,
    beta.pdf(x2plot, 0.7, 2.3),
    color="pink",
    label="imaged planets",
)
# plt.plot(x2plot, beta.pdf(x2plot, 0.7 + 0.3, 2.3 + 0.7), color="pink", alpha=0.2)
# plt.plot(x2plot, beta.pdf(x2plot, 0.7 - 0.3, 2.3 - 0.7), color="pink", alpha=0.2)
# for i in range(n2plot):
#     plt.plot(x2plot, beta.pdf(x2plot, a[i], b[i]), color="pink", alpha=0.2)
plt.fill_between(
    x2plot,
    beta.pdf(x2plot, 0.7 + 0.3, 2.3 + 0.7),
    beta.pdf(x2plot, 0.7 - 0.3, 2.3 - 0.7),
    alpha=0.2,
    color="pink",
)
plt.legend()
plt.xlabel("eccentricity")
plt.ylabel("probability density")
plt.savefig(f"{savedir}/ecc_samples.png", dpi=250)


"""
Corner plot
"""
corner.corner(beta_post, labels=["a", "b"], show_titles=True)
plt.savefig(f"{savedir}/corner.png", dpi=250)
