from ePop import hier_sim
import numpy as np
from scipy.stats import beta
import pandas as pd
import glob
import matplotlib.pyplot as plt


class RVPop_Likelihood(hier_sim.Pop_Likelihood):
    def __init__(
        self, fnames=None, posteriors=None, prior=None, beta_max=100, mu=0.69, std=1.0
    ):
        super().__init__(
            fnames=fnames,
            posteriors=posteriors,
            prior=prior,
            beta_max=beta_max,
            mu=mu,
            std=std,
        )

    def completeness(self, e_array, m=-0.247, b=0.296):
        return m * e_array + b

    def calc_likelihood(self, beta_params):
        """
        This method overwrites ePop!'s default. It makes two main changes wrt the default in ePop!
        1. It assumes that the individual object posteriors were computed with a e^(-1/2) prior applied,
            so it divides this out here to "undo" this.
        2. It corrects for non-uniform completeness, assuming a linear fit to the
            RV eccentricity completeness with m=-.247, b=0.296
        """
        a, b = beta_params

        if a < 0.01 or b < 0.01 or a >= self.beta_max or b >= self.beta_max:
            return -np.inf

        system_sums = np.array(
            [
                np.sum(
                    beta.pdf(ecc_post, a, b)
                    / self.completeness(ecc_post)
                    / ecc_post ** (-0.5)
                )
                / np.shape(ecc_post)[0]
                for ecc_post in self.ecc_posteriors
            ]
        )

        log_likelihood = np.sum(np.log(system_sums))

        log_prior_prob = self.prior.compute_logprob(a, b)

        return log_likelihood + log_prior_prob


ecc_posteriors = []
n_samples = int(
    1e3
)  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results
for post_path in glob.glob("lee_posteriors/*.csv"):
    ecc_post = pd.read_csv(post_path)

    # downsample the posterior to feed into ePop!
    ecc_post = np.random.choice(ecc_post.values.flatten(), size=n_samples)
    ecc_posteriors.append(ecc_post)

n_posteriors = len(ecc_posteriors)
sorted_by_med_idxs = np.flip(
    np.argsort([np.median(ecc_post) for ecc_post in ecc_posteriors])
)

# tower plot
print("Making tower plot!")
fig, ax = plt.subplots(15, 1, figsize=(4, 11), sharex=True)
plt.subplots_adjust(hspace=0)

nbins = 50
for i in np.arange(n_posteriors):
    ax[i].hist(
        ecc_posteriors[sorted_by_med_idxs[i]],
        bins=nbins,
        color="rebeccapurple",
        histtype="step",
        alpha=0.5,
        density=True,
    )
    ax[i].set_yticks([])
    # # overplot the sqrt(e) prior
    # emin = 1 / nbins
    # x2plot = np.linspace(emin, 1, nbins)
    # A = 1 / (2 * (1 - np.sqrt(emin)))  # normalization constant
    # ax[i].plot(x2plot, A / np.sqrt(x2plot), color="rebeccapurple")
plt.xlabel("eccentricity")
plt.savefig("plots/rv_tower_plot.png", dpi=250)

h_prior = None
like = RVPop_Likelihood(posteriors=ecc_posteriors, prior=h_prior)
print("Running MCMC!")
nwalkers = 50
nsteps = 1000
beta_samples = like.sample(nsteps, burn_steps=500, nwalkers=nwalkers)

np.savetxt(
    "plots/{}Prior/epop_samples_{}prior.csv".format(h_prior, h_prior),
    beta_samples,
    delimiter=",",
)
