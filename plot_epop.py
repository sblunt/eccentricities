""""
Plot results from an epop run
"""

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import corner

# read in MCMC samples
hprior = "log-uniform"
oneDcompleteness = False
samples = ["lowmass"]

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

for idx in idx_to_plot:
    a = beta_post[idx, 0]
    b = beta_post[idx, 1]
    plt.plot(x2plot, beta.pdf(x2plot, a, b), color="gold", alpha=0.2)

plt.plot(
    x2plot,
    beta.pdf(x2plot, np.median(beta_post[:, 0]), np.median(beta_post[:, 1])),
    color="k",
    lw=2,
    label="RV planets",
)

# overplot the Kipping result
plt.plot(
    x2plot,
    beta.pdf(x2plot, 0.867, 3.03),
    color="k",
    ls="--",
    lw=2,
    label="Kipping 2013",
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
