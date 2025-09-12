""""
Plot results from an epop run
"""

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import corner

# read in MCMC samples
hprior = "None"
oneDcompleteness = False
samples = ["highmass"]
twocomponent = True
minsma = 0.3

savedir = f"plots/{hprior}Prior"

for sam in samples:
    savedir += f"_{sam}"

savedir += f"minsma{minsma}"

if twocomponent:
    savedir += "_twocomponent"

if oneDcompleteness:
    savedir += "_1Dcompleteness"

beta_post = np.loadtxt(f"{savedir}/epop_samples.csv", delimiter=",")

"""
trend plot
"""
ndim = 2
if twocomponent:
    ndim = 4

fig, ax = plt.subplots(ndim, 1, figsize=(25, 5), sharex=True)
plt.subplots_adjust(hspace=0)
nwalkers = 50

chains = beta_post.reshape((-1, nwalkers, ndim))

for i in range(nwalkers):
    for j in range(ndim):
        ax[j].plot(chains[:, i, j], alpha=0.05, color="k")

plt.savefig(f"{savedir}/trend.png", dpi=250)

"""
Eccentricity samples plot
"""

# pick some samples from the posterior to plot
n2plot = 200
idx_to_plot = np.random.choice(len(beta_post), size=n2plot)

if twocomponent:

    def pdf(x2plot, params):
        a1, b1, a2, b2 = params
        return 0.5 * beta.pdf(x2plot, a1, b1) + 0.5 * beta.pdf(x2plot, a2, b2)

else:

    def pdf(x2plot, params):
        a, b = params
        return beta.pdf(x2plot, a, b)


plt.figure()
x2plot = np.linspace(0, 1, 150)

for idx in idx_to_plot:
    plt.plot(x2plot, pdf(x2plot, beta_post[idx, :]), color="gold", alpha=0.2)

plt.plot(
    x2plot,
    pdf(x2plot, np.median(beta_post, axis=0)),
    # beta.pdf(x2plot, np.median(beta_post[:, 0]), np.median(beta_post[:, 1])),
    color="k",
    lw=2,
    label="{} planets".format(samples[0]),
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
labels = ["a", "b"]

if twocomponent:
    labels = ["a1", "b1", "a2", "b2"]

corner.corner(beta_post, labels=labels, show_titles=True)
plt.savefig(f"{savedir}/corner.png", dpi=250)
