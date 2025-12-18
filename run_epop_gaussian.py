import numpy as np
import glob
import pandas as pd
import os
import emcee
from scipy.stats import norm
from scipy.special import erf
import time


class HierGaussian(object):

    def __init__(
        self,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        n_sma_bins=2,
        n_e_bins=4,
        n_msini_bins=3,
        mass_bin_idx=2,
        sma_bin_idx=1,
    ):
        self.ecc_posteriors = ecc_posteriors
        self.msini_posteriors = msini_posteriors
        self.sma_posteriors = sma_posteriors
        self.mass_posteriors = []

        self.mass_bin_idx = mass_bin_idx
        self.sma_bin_idx = sma_bin_idx

        # read in 3D completeness model
        self.completeness = np.load(
            "completeness_model/{}{}{}completeness.npy".format(
                n_msini_bins, n_e_bins, n_sma_bins
            )
        )

        self.ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
        sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
        sma_bins = sma_bins[sma_bin_idx : sma_bin_idx + 2]
        msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))
        msini_bins = msini_bins[mass_bin_idx : mass_bin_idx + 2]

        # NOTE: here is where we define the bins as uniformly spaced in log(msini) and log(a),
        # and this propagates to the units of our histogram heights
        self.msini_bin_widths = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
        self.sma_bin_widths = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])

        self.ecc_bin_widths = self.ecc_bins[1:] - self.ecc_bins[:-1]

        self.n_e_bins = len(self.ecc_bins) - 1
        self.n_posteriors = len(self.msini_posteriors)

        # in theory the posteriors have different lengths, but I downsample them to all have
        # the same length
        self.post_len = len(self.sma_posteriors[0])
        self.completeness_labels = np.nan * np.ones(
            (self.post_len, 3, self.n_posteriors), dtype=int
        )
        self.mass_labels = np.nan * np.ones(
            (self.post_len, self.n_posteriors), dtype=int
        )

        for k in range(self.n_posteriors):

            for i in range(len(self.ecc_bins) - 1):
                ecc_mask = (self.ecc_posteriors[k] >= self.ecc_bins[i]) & (
                    self.ecc_posteriors[k] < self.ecc_bins[i + 1]
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

                cosi_samples = np.random.uniform(
                    -1, 1, size=len(self.msini_posteriors[k])
                )
                mass_posterior = self.msini_posteriors[k] / (
                    np.sin(np.arccos(cosi_samples))
                )
                self.mass_posteriors.append(mass_posterior)

                mass_mask = (mass_posterior >= msini_bins[i]) & (
                    mass_posterior < msini_bins[i + 1]
                )
                self.mass_labels[mass_mask, k] = i

    def calc_likelihood(self, x):
        """
        This method overwrites ePop!'s default, adding the ability to correct
        for completeness and to fit a pdf that is a gaussian in a single mass/sma bin
        (i.e. the gaussian is only a function of e)

        x: array of size (3) of free parameters: mu, sigma, A
        """

        # apply priors
        mu = x[0]
        sigma = x[1]
        A = x[2]
        if sigma < 0:
            return -np.inf
        if mu < 0 or sigma < 0 or A < 0:
            return -np.inf
        if mu > 1:
            return -np.inf
        if sigma > 1.5:
            return -np.inf
        if A > 100:
            return -np.inf

        def gaussian_val(x):
            return A * np.exp(-(((x - mu) / sigma) ** 2))

        system_sums = np.zeros(self.n_posteriors)
        for i in range(self.n_posteriors):

            for j in range(self.post_len):
                ecc_idx = self.completeness_labels[j, 0, i]
                sma_idx = self.completeness_labels[j, 1, i]
                msini_idx = self.completeness_labels[j, 2, i]
                mass_idx = self.mass_labels[j, i]

                if not np.isnan(ecc_idx + sma_idx + mass_idx + msini_idx):

                    ecc_idx = int(ecc_idx)
                    sma_idx = int(sma_idx)
                    msini_idx = int(msini_idx)
                    mass_idx = int(mass_idx)

                    system_sums[i] += (
                        self.completeness[ecc_idx, sma_idx, msini_idx]
                        * gaussian_val(self.ecc_posteriors[i][j])
                        / self.post_len
                    )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        # add in exponential part of HBM likelihood
        # this is (negative) the expected number of planets detected by the survey; good sanity check
        def gaussian_integral(x, mu, sigma):
            return 0.5 * np.sqrt(np.pi) * sigma * erf((mu - x) / sigma)

        norm_constant = 0
        for i in np.arange(self.n_e_bins):
            out_of_integral_constant = (
                A
                * self.msini_bin_widths
                * self.sma_bin_widths
                * self.completeness[i, self.sma_bin_idx, self.mass_bin_idx]
            )

            d_norm_constant = out_of_integral_constant * (
                gaussian_integral(self.ecc_bins[i], mu, sigma)
                - gaussian_integral(self.ecc_bins[i + 1], mu, sigma)
            )
            norm_constant -= d_norm_constant
        log_likelihood += norm_constant

        return log_likelihood

    def sample(self, nsteps, burn_steps=200, nwalkers=100):

        ndim = 3
        p0 = np.random.uniform(0, 1, size=(nwalkers, ndim))
        p0[:, 2] = np.random.uniform(0, 100, size=nwalkers)

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
    n_samples = 50  # 999 # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results

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

        # downsample the posteriors
        idxs = np.random.choice(np.arange(post_len), size=n_samples, replace=False)

        ecc_posteriors.append(ecc_post[idxs])
        msini_posteriors.append(msini_post[idxs])
        sma_posteriors.append(sma_post[idxs])

    n_msini_bins = 3
    n_sma_bins = 1
    n_e_bins = 5
    mass_idx = 1
    sma_idx = 0

    like = HierGaussian(
        ecc_posteriors,
        msini_posteriors=msini_posteriors,
        sma_posteriors=sma_posteriors,
        n_sma_bins=n_sma_bins,
        n_e_bins=n_e_bins,
        n_msini_bins=n_msini_bins,
        mass_bin_idx=mass_idx,
        sma_bin_idx=sma_idx,
    )

    print("Running MCMC!")
    burn_steps = 500
    nwalkers = 100
    nsteps = 500

    hbm_samples = like.sample(
        nsteps,
        burn_steps=burn_steps,
        nwalkers=nwalkers,
    )

    savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    np.savetxt(
        "{}/gaussian_samples_burn{}_total{}_massidx{}_smaidx{}.csv".format(
            savedir, burn_steps, nsteps, mass_idx, sma_idx
        ),
        hbm_samples,
        delimiter=",",
    )
