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
        n_sma_bins=4,
        n_e_bins=4,
        n_msini_bins=2,
    ):
        self.ecc_posteriors = ecc_posteriors
        self.msini_posteriors = msini_posteriors
        self.sma_posteriors = sma_posteriors
        self.mass_posteriors = []

        # read in 3D completeness model
        self.completeness = np.load(
            "completeness_model/{}{}{}completeness.npy".format(
                n_msini_bins, n_e_bins, n_sma_bins
            )
        )
        self.ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
        sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
        msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))

        # NOTE: here is where we define the bins as uniformly spaced in log(msini) and log(a),
        # and this propagates to the units of our histogram heights
        self.msini_bin_widths = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
        self.sma_bin_widths = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])
        self.ecc_bin_widths = self.ecc_bins[1:] - self.ecc_bins[:-1]

        self.n_e_bins = len(self.ecc_bins) - 1
        self.n_sma_bins = len(sma_bins) - 1
        self.n_msini_bins = len(msini_bins) - 1

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
                        * histogram_heights[ecc_idx, sma_idx, mass_idx]
                        / self.post_len
                    )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        # add in exponential part of HBM likelihood
        # this is (negative) the expected number of planets detected by the survey; good sanity check
        norm_constant = -np.sum(self.completeness * histogram_heights * self.bin_widths)
        # print(norm_constant)
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
    n_samples = 999  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results

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

    n_msini_bins = 2
    n_sma_bins = 2
    n_e_bins = 3

    like = HierHistogram(
        ecc_posteriors,
        msini_posteriors=msini_posteriors,
        sma_posteriors=sma_posteriors,
        n_sma_bins=n_sma_bins,
        n_e_bins=n_e_bins,
        n_msini_bins=n_msini_bins,
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
        "{}/epop_samples_burn{}_total{}.csv".format(savedir, burn_steps, nsteps),
        hbm_samples,
        delimiter=",",
    )
