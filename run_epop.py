from ePop import hier_sim
import numpy as np
from scipy.stats import beta
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# TODO: take arguments about msini/projected separation into account in sample selection
# TODO: jason suggests expanding sample size in mass/sma until we can get a strongly-constrained posterior
# TODO: interpretation: could it be an age effect or a stellar mass effect?


class RVPop_Likelihood(hier_sim.Pop_Likelihood):
    def __init__(
        self,
        fnames=None,
        ecc_posteriors=None,
        K_posteriors=None,
        per_posteriors=None,
        prior=None,
        beta_max=100,
        mu=0.69,
        std=1.0,
        oneD_completeness=False,
    ):
        super().__init__(
            fnames=fnames,
            posteriors=ecc_posteriors,
            prior=prior,
            beta_max=beta_max,
            mu=mu,
            std=std,
        )

        completeness_ebins = np.load("e_bins.npy")
        completeness_Kbins = np.load("K_bins.npy")
        completeness_perbins = np.load("per_bins.npy")
        completeness_map = np.load("completeness.npy")

        self.completeness = {}
        for i, e_array in enumerate(self.ecc_posteriors):

            closest_e_indices = np.argmin(
                np.abs(e_array[:, None] - completeness_ebins), axis=1
            )
            K_array = K_posteriors[i]
            closest_K_indices = np.argmin(
                np.abs(K_array[:, None] - completeness_Kbins), axis=1
            )
            per_array = per_posteriors[i]
            closest_per_indices = np.argmin(
                np.abs(per_array[:, None] - completeness_perbins), axis=1
            )

            self.completeness[i] = completeness_map[
                closest_e_indices, closest_per_indices, closest_K_indices
            ]
        self.apply_oneD_completeness = oneD_completeness

    def oneD_completeness(self, e_array, m=-0.247, b=0.296):
        """
        Returns the fraction of systems that are observable for a given eccentricity array
        """
        return m * e_array + b

    def calc_likelihood(self, beta_params):
        """
        This method overwrites ePop!'s default, adding the ability to correct
        for completeness.
        """
        a, b = beta_params

        if a < 0.01 or b < 0.01 or a >= self.beta_max or b >= self.beta_max:
            return -np.inf

        if self.apply_oneD_completeness:
            system_sums = np.array(
                [
                    np.sum(beta.pdf(ecc_post, a, b) / self.oneD_completeness(ecc_post))
                    / np.shape(ecc_post)[0]
                    for ecc_post in self.ecc_posteriors
                ]
            )
        else:
            system_sums = np.array(
                [
                    np.sum(beta.pdf(ecc_post, a, b) / self.completeness[i])
                    / np.shape(ecc_post)[0]
                    for i, ecc_post in enumerate(self.ecc_posteriors)
                ]
            )

        log_likelihood = np.sum(np.log(system_sums))

        log_prior_prob = self.prior.compute_logprob(a, b)

        return log_likelihood + log_prior_prob


ecc_posteriors = []
K_posteriors = []
per_posteriors = []
n_samples = int(
    1e3
)  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results
for post_path in glob.glob("lee_posteriors/ecc_*.csv"):
    ecc_post = pd.read_csv(post_path)

    # downsample the posterior to feed into ePop!
    ecc_post = np.random.choice(ecc_post.values.flatten(), size=n_samples)
    ecc_posteriors.append(ecc_post)

for post_path in glob.glob("lee_posteriors/K_*.csv"):
    K_post = pd.read_csv(post_path)

    # downsample the posterior to feed into ePop!
    K_post = np.random.choice(K_post.values.flatten(), size=n_samples)
    K_posteriors.append(K_post)

for post_path in glob.glob("lee_posteriors/per_*.csv"):
    per_post = pd.read_csv(post_path)

    # downsample the posterior to feed into ePop!
    per_post = np.random.choice(per_post.values.flatten(), size=n_samples)
    per_posteriors.append(per_post)

n_posteriors = len(ecc_posteriors)
sorted_by_med_idxs = np.flip(
    np.argsort([np.median(ecc_post) for ecc_post in ecc_posteriors])
)

# tower plot
make_tower_plot = False
if make_tower_plot:
    print("Making tower plot!")
    fig, ax = plt.subplots(15, 3, figsize=(4, 11))
    plt.subplots_adjust(hspace=0)

    nbins = 50
    for i in np.arange(n_posteriors):
        ax[i, 0].hist(
            ecc_posteriors[sorted_by_med_idxs[i]],
            bins=nbins,
            color="rebeccapurple",
            histtype="step",
            alpha=0.5,
            density=True,
        )
        ax[i, 0].set_yticks([])
        ax[i, 0].set_xlim(0, 1)
        ax[i, 1].hist(
            K_posteriors[sorted_by_med_idxs[i]],
            bins=nbins,
            color="rebeccapurple",
            histtype="step",
            alpha=0.5,
            density=True,
            range=(0, 1000),
        )
        ax[i, 1].set_yticks([])
        ax[i, 1].set_xlim(1, 1000)

        ax[i, 2].hist(
            per_posteriors[sorted_by_med_idxs[i]],
            bins=nbins,
            color="rebeccapurple",
            histtype="step",
            alpha=0.5,
            density=True,
            range=(0, 1e5),
        )
        # # overplot the sqrt(e) prior
        # emin = 1 / nbins
        # x2plot = np.linspace(emin, 1, nbins)
        # A = 1 / (2 * (1 - np.sqrt(emin)))  # normalization constant
        # ax[i].plot(x2plot, A / np.sqrt(x2plot), color="rebeccapurple")
    ax[-1, 0].set_xlabel("eccentricity")
    ax[-1, 1].set_xlabel("K [m/s]")
    plt.savefig("plots/rv_tower_plot.png", dpi=250)

h_prior = "gaussian"
oneD_completeness = True
like = RVPop_Likelihood(
    ecc_posteriors=ecc_posteriors,
    K_posteriors=K_posteriors,
    per_posteriors=per_posteriors,
    prior=h_prior,
    oneD_completeness=oneD_completeness,
)
print("Running MCMC!")
burn_steps = 500
nwalkers = 50
nsteps = 200
beta_samples = like.sample(nsteps, burn_steps=burn_steps, nwalkers=nwalkers)

savedir = f"plots/{h_prior}Prior"
if oneD_completeness:
    savedir += "_1Dcompleteness"
if not os.path.exists(savedir):
    os.mkdir(savedir)

np.savetxt(
    "{}/epop_samples.csv".format(savedir),
    beta_samples,
    delimiter=",",
)
