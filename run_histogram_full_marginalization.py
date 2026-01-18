import numpy as np
import glob
import pandas as pd
import os
import emcee
from functools import lru_cache

# proper incliantion marginalization

num_stars_cps = 719


class HierHistogram(object):

    def __init__(
        self,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        n_sma_bins=4,
        n_e_bins=4,
        n_msini_bins=2,
        bd_prior=True,
    ):
        self.ecc_posteriors = ecc_posteriors
        self.msini_posteriors = msini_posteriors
        self.sma_posteriors = sma_posteriors
        self.bd_prior = bd_prior

        # read in 3D completeness model
        self.completeness = np.load(
            "completeness_model/{}{}{}completeness.npy".format(
                n_msini_bins, n_e_bins, n_sma_bins
            )
        )
        self.ecc_bin_edges = np.load(
            "completeness_model/{}ecc_bins.npy".format(n_e_bins)
        )
        sma_bin_edges = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
        msini_bin_edges = np.load(
            "completeness_model/{}msini_bins.npy".format(n_msini_bins)
        )
        self.msini_bin_edges = msini_bin_edges

        self.mass_bin_edges = np.copy(msini_bin_edges)

        # NOTE: here is where we define the bins as uniformly spaced in log(msini) and log(a),
        # and this propagates to the units of our histogram heights
        self.msini_bin_widths = np.log(self.msini_bin_edges[1:]) - np.log(
            self.msini_bin_edges[:-1]
        )

        self.sma_bin_widths = np.log(sma_bin_edges[1:]) - np.log(sma_bin_edges[:-1])
        self.ecc_bin_widths = self.ecc_bin_edges[1:] - self.ecc_bin_edges[:-1]
        self.mass_bin_widths = np.log(self.mass_bin_edges[1:]) - np.log(
            self.mass_bin_edges[:-1]
        )

        self.n_e_bins = len(self.ecc_bin_edges) - 1
        self.n_sma_bins = len(sma_bin_edges) - 1
        self.n_msini_bins = len(msini_bin_edges) - 1
        self.n_mass_bins = len(self.mass_bin_edges) - 1

        self.n_posteriors = len(self.msini_posteriors)

        # in theory the posteriors could have different lengths, but I downsample them to all have
        # the same length in pre-processing
        self.post_len = len(self.sma_posteriors[0])
        self.completeness_labels = np.nan * np.ones(
            (self.post_len, 3, self.n_posteriors), dtype=int
        )

        for k in range(self.n_posteriors):

            for i in range(self.n_e_bins):
                ecc_mask = (self.ecc_posteriors[k] >= self.ecc_bin_edges[i]) & (
                    self.ecc_posteriors[k] < self.ecc_bin_edges[i + 1]
                )
                self.completeness_labels[ecc_mask, 0, k] = i
            for i in range(self.n_sma_bins):
                sma_mask = (self.sma_posteriors[k] >= sma_bin_edges[i]) & (
                    self.sma_posteriors[k] < sma_bin_edges[i + 1]
                )
                self.completeness_labels[sma_mask, 1, k] = i
            for i in range(self.n_msini_bins):
                msini_mask = (self.msini_posteriors[k] >= self.msini_bin_edges[i]) & (
                    self.msini_posteriors[k] < self.msini_bin_edges[i + 1]
                )
                self.completeness_labels[msini_mask, 2, k] = i

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
            (self.n_e_bins, self.n_sma_bins, self.n_mass_bins)
        )

        bd_occurrence = 0
        bd_max_occurrence = 0.05
        if (
            self.bd_prior
        ):  # set a prior that keeps bd occurrence rate below bd_max_occurrence
            # (note: this didn't end up being necessary-- the bd desert occurrence rate
            # appeared to be ~2%, and constrained by the data. This may be contaminated
            # by stellar binaries though.)
            for i in np.arange(self.n_e_bins):
                for j in np.arange(self.n_sma_bins):
                    bd_occurrence += (
                        histogram_heights[i, j, -1]
                        * self.ecc_bin_widths[i]
                        * self.sma_bin_widths[j]
                        * self.mass_bin_widths[-1]
                    ) / num_stars_cps

            if bd_occurrence > bd_max_occurrence:
                return -np.inf

        system_sums = np.zeros(self.n_posteriors)
        for i in range(self.n_posteriors):

            for j in range(self.post_len):

                ecc_idx = self.completeness_labels[j, 0, i]
                sma_idx = self.completeness_labels[j, 1, i]
                msini_idx = self.completeness_labels[j, 2, i]

                if not np.isnan(msini_idx) and not np.isnan(sma_idx):
                    ecc_idx = int(ecc_idx)
                    sma_idx = int(sma_idx)
                    msini_idx = int(msini_idx)

                    msini_value = self.msini_posteriors[i][j]
                    for mass_idx in np.arange(self.n_mass_bins):

                        mass_val_hi = self.mass_bin_edges[mass_idx + 1]
                        msini_m_hi = msini_value / mass_val_hi
                        if msini_m_hi > 1:
                            msini_m_hi = 1

                        mass_val_lo = self.mass_bin_edges[mass_idx]
                        msini_m_lo = msini_value / mass_val_lo
                        if msini_m_lo > 1:
                            msini_m_lo = 1

                        incweight_factor = np.sqrt(1 - msini_m_hi**2) - np.sqrt(
                            1 - msini_m_lo**2
                        )  # can sanity check here that these add up to 1:
                        # print(incweight_factor)

                        system_sum_j = (
                            self.completeness[ecc_idx, sma_idx, msini_idx]
                            * (
                                histogram_heights[
                                    ecc_idx, sma_idx, mass_idx
                                ]  # basically we're weighting each histogram height (occurrence rate in mass space) by the fraction of inclination probability space
                                * incweight_factor
                            )
                            / self.post_len
                        )
                        system_sums[i] += system_sum_j

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        @lru_cache
        def _integral(x, A):
            """integral of sqrt(1- ((e^x) / e^A)^2 )

            x is used for log(msini), A is log(mass)
            """
            if x - A > 0:  # if msini < mass
                return 0
            else:

                const = np.sqrt(1 - np.exp(2 * (x - A)))

                return const - np.arctanh(const)

        @lru_cache
        def msini_integral(mass_idx, msini_bin_idx):
            msini_hi = self.msini_bin_edges[msini_bin_idx + 1]
            msini_lo = self.msini_bin_edges[msini_bin_idx]

            mass_hi = self.mass_bin_edges[mass_idx + 1]
            mass_lo = self.mass_bin_edges[mass_idx]

            msini_integral = (
                (
                    _integral(np.log(msini_hi), np.log(mass_hi))
                    - _integral(np.log(msini_lo), np.log(mass_hi))
                )
            ) - (
                _integral(np.log(msini_hi), np.log(mass_lo))
                - _integral(np.log(msini_lo), np.log(mass_lo))
            )

            return msini_integral

        norm_constant = 0
        # integrate mass and inc over msini boundaries, multiplying at each step by completeness and a/e integral
        for ecc_idx in np.arange(self.n_e_bins):
            for sma_idx in np.arange(self.n_sma_bins):
                for msini_bin_idx in np.arange(self.n_msini_bins):
                    for mass_idx in np.arange(self.n_mass_bins):

                        norm_constant += (
                            self.completeness[ecc_idx, sma_idx, msini_bin_idx]
                            * msini_integral(
                                mass_idx, msini_bin_idx
                            )  # NOTE: this converts dG/dMdi to dG/dMsini
                            * histogram_heights[
                                ecc_idx, sma_idx, mass_idx
                            ]  # NOTE: could vectorize this for speed boost
                            * self.ecc_bin_widths[ecc_idx]
                            * self.sma_bin_widths[sma_idx]
                        )
        # this is the expected total number of planets detected; good sanity check
        print(norm_constant)
        log_likelihood -= norm_constant

        return log_likelihood

    def sample(self, nsteps, burn_steps=200, nwalkers=100):

        ndim = self.n_e_bins * self.n_sma_bins * self.n_mass_bins
        p0 = np.random.uniform(
            0, 25, size=(nwalkers, self.n_e_bins, self.n_sma_bins, self.n_mass_bins)
        )

        # start the bd occurrence rates lower to ease convergence
        p0[:, :, :, -1] = np.random.uniform(
            0, 5, size=(nwalkers, self.n_e_bins, self.n_sma_bins)
        )

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calc_likelihood)
        state = sampler.run_mcmc(
            p0.reshape((nwalkers, ndim)), burn_steps, progress=True
        )

        print("Burn in complete!")

        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=True)

        posterior = sampler.get_chain(flat=True)

        return posterior


if __name__ == "__main__":

    ecc_posteriors = []
    msini_posteriors = []
    sma_posteriors = []
    n_samples = 50  # 999  # according to Hogg+ '10 paper, you can go as low as 50 samples per posterior and get reasonable results

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

    savedir = f"plots/fullmarg_bdprior_{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    np.savetxt(
        "{}/epop_samples_burn{}_total{}.csv".format(savedir, burn_steps, nsteps),
        hbm_samples,
        delimiter=",",
    )
