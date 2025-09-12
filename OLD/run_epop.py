from ePop import hier_sim
import numpy as np
from scipy.stats import beta
import pandas as pd
import glob
import os


class RVPop_Likelihood(hier_sim.Pop_Likelihood):
    def __init__(
        self,
        fnames=None,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        prior=None,
        beta_max=100,
        mu=0.69,
        std=1.0,
        oneD_completeness=False,
        min_sma=1,
    ):
        # min_sma: minimum semimajor axis (au) to consider in the analysis. I set
        # the completeness values for any posterior samples with smaller smas = 0.
        # This is basically a hack to play with sample selection.

        super().__init__(
            fnames=fnames,
            posteriors=ecc_posteriors,
            prior=prior,
            beta_max=beta_max,
            mu=mu,
            std=std,
        )

        self.msini_posteriors = msini_posteriors
        self.sma_posteriors = sma_posteriors

        self.apply_oneD_completeness = oneD_completeness

        # read in 3D completeness model
        completeness = np.load("completeness_model/completeness.npy")
        ecc_bins = np.load("completeness_model/ecc_bins.npy")
        sma_bins = np.load("completeness_model/sma_bins.npy")
        msini_bins = np.load("completeness_model/msini_bins.npy")

        # assign completeness indices to each posterior sample
        """
        Returns the fraction of systems that are observable for a given ecc/msini/sma array,
        accounting for covariances between the three. Completeness is calculated by directly
        adding up BJ's recoveries and dividing by his injections, then using linear
        interpolation to fill in the gaps in the 3d parameter space. The values
        are calculated in make_frelikh_comparison_plot.py. Strictly speaking, BJ did
        different numbers of injections as a function of eccentricity, so the completeness
        values at, e.g., higher eccentricity should have higher uncertainties.
        TODO: think about this more. Calculate typical uncertainty in completeness using Poisson
        and decide if it matters.
        """
        n_posteriors = len(self.msini_posteriors)
        self.completeness = []

        for k in range(n_posteriors):
            post_len = len(self.sma_posteriors[k])
            completeness_labels_i = 1000 * np.ones((post_len, 3), dtype=int)

            for i in range(len(ecc_bins) - 1):
                ecc_mask = (self.ecc_posteriors[k] >= ecc_bins[i]) & (
                    self.ecc_posteriors[k] < ecc_bins[i + 1]
                )
                completeness_labels_i[ecc_mask, 0] = i
            for i in range(len(sma_bins) - 1):
                sma_mask = (self.sma_posteriors[k] >= sma_bins[i]) & (
                    self.sma_posteriors[k] < sma_bins[i + 1]
                )
                completeness_labels_i[sma_mask, 1] = i
            for i in range(len(msini_bins) - 1):
                msini_mask = (self.msini_posteriors[k] >= msini_bins[i]) & (
                    self.msini_posteriors[k] < msini_bins[i + 1]
                )
                completeness_labels_i[msini_mask, 2] = i

            completeness_post_i = np.zeros(post_len)
            for i in range(post_len):
                a, b, c = completeness_labels_i[i]
                if a + b + c < 100:
                    completeness_post_i[i] = completeness[a, b, c]
                else:
                    completeness_post_i[i] = 0

            # RULES OUT SMAs SMALLER THAN A CERTAIN VALUE (ie. changes sample selection)
            completeness_post_i[self.sma_posteriors[k] < min_sma] = 0

            self.completeness.append(completeness_post_i)

    def oneD_completeness(self, e_array, m=-0.325, b=0.348):
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
                    np.sum(
                        np.nan_to_num(
                            beta.pdf(ecc_post, a, b) / self.completeness[i],
                            posinf=0.0,
                            nan=0.0,
                        )
                    )
                    / np.shape(ecc_post)[0]
                    for i, ecc_post in enumerate(self.ecc_posteriors)
                ]
            )

        # TODO: I added a nansum above because some completeness values are 0. This
        # effectively removes posterior samples that are in parts of parameter
        # space the survey is 0% complete to. I think this makes sense, but think about this a little more.

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        log_prior_prob = self.prior.compute_logprob(a, b)

        return log_likelihood + log_prior_prob


class TwoBetaPop_Likelihood(RVPop_Likelihood):
    def __init__(
        self,
        fnames=None,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        prior=None,
        beta_max=100,
        mu=0.69,
        std=1.0,
        min_sma=1,
    ):
        super().__init__(
            fnames=fnames,
            ecc_posteriors=ecc_posteriors,
            msini_posteriors=msini_posteriors,
            sma_posteriors=sma_posteriors,
            prior=prior,
            beta_max=beta_max,
            mu=mu,
            std=std,
            oneD_completeness=False,
            min_sma=min_sma,
        )

    def calc_likelihood(self, beta_params):
        """
        This method overwrites ePop!'s default, adding the ability to correct
        for completeness and to fit a pdf that is the sum of two gaussians
        """
        a1, b1, a2, b2 = beta_params

        if a1 < 0.01 or b1 < 0.01 or a1 >= self.beta_max or b1 >= self.beta_max:
            return -np.inf
        if a2 < 0.01 or b2 < 0.01 or a2 >= self.beta_max or b2 >= self.beta_max:
            return -np.inf

        system_sums = np.array(
            [
                np.sum(
                    np.nan_to_num(
                        (
                            0.5 * beta.pdf(ecc_post, a1, b1)
                            + 0.5 * beta.pdf(ecc_post, a2, b2)
                        )
                        / self.completeness[i],
                        posinf=0.0,
                        nan=0.0,
                    )
                )
                / np.shape(ecc_post)[0]
                for i, ecc_post in enumerate(self.ecc_posteriors)
            ]
        )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        log_prior1_prob = self.prior.compute_logprob(a1, b1)
        log_prior2_prob = self.prior.compute_logprob(a2, b2)

        return log_likelihood + log_prior1_prob + log_prior2_prob


ecc_posteriors = []
msini_posteriors = []
sma_posteriors = []
n_samples = int(
    1e3
)  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results

samples = ["highmass"]  # , "highmass"]
h_prior = None
oneD_completeness = False
twocomponent = True
min_sma = 0.3

for sam in samples:
    for post_path in glob.glob("lee_posteriors/{}/ecc_*.csv".format(sam)):
        ecc_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        ecc_post = np.random.choice(ecc_post.values.flatten(), size=n_samples)
        ecc_posteriors.append(ecc_post)

    for post_path in glob.glob("lee_posteriors/{}/msini_*.csv".format(sam)):
        msini_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        msini_post = np.random.choice(msini_post.values.flatten(), size=n_samples)
        msini_posteriors.append(msini_post)

    for post_path in glob.glob("lee_posteriors/{}/sma_*.csv".format(sam)):
        sma_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        sma_post = np.random.choice(sma_post.values.flatten(), size=n_samples)
        sma_posteriors.append(sma_post)

n_posteriors = len(ecc_posteriors)

if twocomponent:
    like = TwoBetaPop_Likelihood(
        ecc_posteriors=ecc_posteriors,
        sma_posteriors=sma_posteriors,
        msini_posteriors=msini_posteriors,
        prior=h_prior,
    )
else:
    like = RVPop_Likelihood(
        ecc_posteriors=ecc_posteriors,
        sma_posteriors=sma_posteriors,
        msini_posteriors=msini_posteriors,
        prior=h_prior,
        oneD_completeness=oneD_completeness,
    )
print("Running MCMC!")
burn_steps = 200
nwalkers = 50
nsteps = 200

ndim = 2
if twocomponent:
    ndim = 4

beta_samples = like.sample(nsteps, burn_steps=burn_steps, nwalkers=nwalkers, ndim=ndim)

savedir = f"plots/{h_prior}Prior"

for sam in samples:
    savedir += f"_{sam}"

savedir += "minsma{}".format(min_sma)

if twocomponent:
    savedir += "_twocomponent"

if oneD_completeness:
    savedir += "_1Dcompleteness"

if not os.path.exists(savedir):
    os.mkdir(savedir)

np.savetxt(
    "{}/epop_samples.csv".format(savedir),
    beta_samples,
    delimiter=",",
)
