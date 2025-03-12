import numpy as np
import glob
import pandas as pd
import os
import emcee


class HierHistogram(object):

    def __init__(
        self,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        sma_priors=None,
        msini_priors=None,
        n_sma_bins=4,
        n_e_bins=4,
        n_msini_bins=2,
    ):
        self.ecc_posteriors = ecc_posteriors
        self.msini_posteriors = msini_posteriors
        self.sma_posteriors = sma_posteriors
        self.sma_priors = sma_priors
        self.msini_priors = msini_priors

        # read in 3D completeness model
        self.completeness = np.load(
            "completeness_model/{}{}{}completeness.npy".format(
                n_msini_bins, n_e_bins, n_sma_bins
            )
        )
        ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
        sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
        msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))

        # NOTE: here is where we define the bins as uniformly spaced in log(msini) and log(a),
        # and this propagates to the units of our histogram heights
        self.msini_bin_widths = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
        self.sma_bin_widths = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])
        self.ecc_bin_widths = ecc_bins[1:] - ecc_bins[:-1]

        self.n_e_bins = len(ecc_bins) - 1
        self.n_sma_bins = len(sma_bins) - 1
        self.n_msini_bins = len(msini_bins) - 1

        self.n_posteriors = len(self.msini_posteriors)

        # in theory the posteriors have different lengths, but I downsample them to all have
        # the same length
        self.post_len = len(self.sma_posteriors[0])
        self.completeness_labels = np.nan * np.ones(
            (self.post_len, 3, self.n_posteriors), dtype=int
        )

        for k in range(self.n_posteriors):

            for i in range(len(ecc_bins) - 1):
                ecc_mask = (self.ecc_posteriors[k] >= ecc_bins[i]) & (
                    self.ecc_posteriors[k] < ecc_bins[i + 1]
                )
                self.completeness_labels[ecc_mask, 0, k] = i
            for i in range(len(sma_bins) - 1):
                sma_mask = (self.sma_posteriors[k] >= sma_bins[i]) & (
                    self.sma_posteriors[k] < sma_bins[i + 1]
                )
                self.completeness_labels[sma_mask, 1, k] = i
            for i in range(len(msini_bins) - 1):
                msini_mask = (self.msini_posteriors[k] >= msini_bins[i]) & (
                    self.msini_posteriors[k] < msini_bins[i + 1]
                )
                self.completeness_labels[msini_mask, 2, k] = i

        self.bin_widths = np.zeros((self.n_e_bins, self.n_sma_bins, self.n_msini_bins))
        for i in range(self.n_e_bins):
            for j in range(self.n_sma_bins):
                for k in range(self.n_msini_bins):
                    self.bin_widths[i, j, k] = (
                        self.ecc_bin_widths[i]
                        * self.sma_bin_widths[j]
                        * self.msini_bin_widths[k]
                    )

    def calc_likelihood(self, x):
        """
        This method overwrites ePop!'s default, adding the ability to correct
        for completeness and to fit a pdf that is just histogram heights in a,e, msini
        space

        histogram_heights: array of size (N_ecc x N_a x N_msini) of free parameters
        """

        # apply priors keeping histogram heights above 0
        for i in x:
            if i < 0:
                return -np.inf
        histogram_heights = x.reshape(
            (self.n_e_bins, self.n_sma_bins, self.n_msini_bins)
        )
        # NOTE: effective priors radvel applied to individual planet posteriors are computed
        # numerically (except on e, which is uniform) in get_posteriors.py

        system_sums = np.zeros(self.n_posteriors)
        for i in range(self.n_posteriors):

            for j in range(self.post_len):
                ecc_idx = self.completeness_labels[j, 0, i]
                sma_idx = self.completeness_labels[j, 1, i]
                msini_idx = self.completeness_labels[j, 2, i]
                if not np.isnan(ecc_idx + sma_idx + msini_idx):
                    ecc_idx = int(ecc_idx)
                    sma_idx = int(sma_idx)
                    msini_idx = int(msini_idx)

                    system_sums[i] += (
                        self.completeness[ecc_idx, sma_idx, msini_idx]
                        * histogram_heights[ecc_idx, sma_idx, msini_idx]
                        / self.post_len  # TODO: removed prior correction for now; add it back in and see how much it changes the result
                        # / (
                        #     self.msini_priors[i][msini_idx]
                        #     * self.sma_priors[i][sma_idx]
                        # )
                    )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        # add in exponential part of HBM likelihood
        # this is (negative) the expected number of planets detected by the survey; good sanity check
        norm_constant = -np.sum(self.completeness * histogram_heights * self.bin_widths)
        print(norm_constant)
        log_likelihood += norm_constant

        return log_likelihood

    def sample(self, nsteps, burn_steps=200, nwalkers=100):

        ndim = self.n_e_bins * self.n_sma_bins * self.n_msini_bins
        p0 = np.random.uniform(0, 50, size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calc_likelihood)
        state = sampler.run_mcmc(p0, burn_steps, progress=True)

        print("Burn in complete!")

        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=True)

        posterior = sampler.get_chain(flat=True)

        return posterior


if __name__ == "__main__":

    ecc_posteriors = []
    msini_posteriors = []
    sma_posteriors = []
    msini_priors = []
    sma_priors = []
    n_samples = int(
        1e2  # TODO: change back to 1e3 for final  if needed
    )  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results
    print("reading e posteriors...")

    for post_path in glob.glob("lee_posteriors/*/ecc_*.csv"):
        ecc_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        ecc_post = np.random.choice(ecc_post.values.flatten(), size=n_samples)
        ecc_posteriors.append(ecc_post)

    print("reading msini posteriors...")
    for post_path in glob.glob("lee_posteriors/*/msini_*.csv"):
        msini_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        msini_post = np.random.choice(msini_post.values.flatten(), size=n_samples)
        msini_posteriors.append(msini_post)
    print("reading sma posteriors...")

    for post_path in glob.glob("lee_posteriors/*/sma_*.csv"):
        sma_post = pd.read_csv(post_path)

        # downsample the posterior to feed into ePop!
        sma_post = np.random.choice(sma_post.values.flatten(), size=n_samples)
        sma_posteriors.append(sma_post)
    print("reading sma priors...")

    for post_path in glob.glob("lee_posteriors/*/smaPRIOR*.csv"):
        sma_prior = pd.read_csv(post_path)

        prior_hist, bins = np.histogram(sma_prior, bins=50, density=True)

        # figure out which prior bins the posterior samples fall into
        sma_prior_probs = np.zeros(len(sma_post))
        for i, a_i in enumerate(sma_post):
            for j in np.arange(len(bins) - 1):
                if a_i > bins[j] and a_i <= bins[j + 1]:
                    sma_prior_probs[i] = prior_hist[j]
        sma_priors.append(sma_prior_probs)
    print("reading msini priors...")

    for post_path in glob.glob("lee_posteriors/*/msiniPRIOR*.csv"):
        msini_prior = pd.read_csv(post_path)

        prior_hist, bins = np.histogram(msini_prior, bins=50, density=True)

        # figure out which prior bins the posterior samples fall into
        msini_prior_probs = np.zeros(len(msini_post))
        for i, m_i in enumerate(msini_post):
            for j in np.arange(len(bins) - 1):
                if m_i > bins[j] and m_i <= bins[j + 1]:
                    msini_prior_probs[i] = prior_hist[j]
        msini_priors.append(msini_prior_probs)

    n_msini_bins = 2
    n_sma_bins = 6
    n_e_bins = 1

    like = HierHistogram(
        ecc_posteriors,
        msini_posteriors=msini_posteriors,
        sma_posteriors=sma_posteriors,
        msini_priors=msini_priors,
        sma_priors=sma_priors,
        n_sma_bins=n_sma_bins,
        n_e_bins=n_e_bins,
        n_msini_bins=n_msini_bins,
    )

    print("Running MCMC!")
    burn_steps = 500
    nwalkers = 100
    nsteps = 200

    hbm_samples = like.sample(
        nsteps,
        burn_steps=burn_steps,
        nwalkers=nwalkers,
    )

    savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    np.savetxt(
        "{}/epop_samples_burn{}_total{}.csv".format(savedir, burn_steps, nsteps),
        hbm_samples,
        delimiter=",",
    )
