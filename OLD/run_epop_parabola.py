import numpy as np
import glob
import pandas as pd
import os
import emcee
from run_epop_histogram import HierHistogram


class HierParab(HierHistogram):

    def __init__(
        self,
        ecc_posteriors=None,
        msini_posteriors=None,
        sma_posteriors=None,
        n_sma_bins=4,
        n_e_bins=4,
        n_msini_bins=2,
    ):
        super().__init__(
            ecc_posteriors=ecc_posteriors,
            msini_posteriors=msini_posteriors,
            sma_posteriors=sma_posteriors,
            n_sma_bins=n_sma_bins,
            n_e_bins=n_e_bins,
            n_msini_bins=n_msini_bins,
            highsmaonly=True,
        )

        assert (
            self.n_sma_bins == 1
        )  # (I'm defining the likelihood fn below assuming this is true)

    def calc_likelihood(self, x):
        """
        Correct for completeness and fit a parabola as a fn of eccentricity
        to each msini range
        """

        As = np.empty(self.n_msini_bins)
        Bs = np.empty(self.n_msini_bins)
        Cs = np.empty(self.n_msini_bins)
        for i in range(self.n_msini_bins):
            As[i] = x[3 * i]
            Bs[i] = x[3 * i + 1]
            Cs[i] = x[3 * i + 2]

            # set priors that keep occurrence >0 on range 0-1
            if Cs[i] < 0:  # >0 at 0
                return -np.inf
            if As[i] ** 2 + Bs[i] + Cs[i] < 0:  # >0 at 1
                return -np.inf

        def parab(a, b, c, ecc):
            return a * ecc**2 + b * ecc + c

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

                    ecc_val = self.ecc_posteriors[i][j]

                    system_sums[i] += (
                        self.completeness[ecc_idx, sma_idx, msini_idx]
                        * parab(As[msini_idx], Bs[msini_idx], Cs[msini_idx], ecc_val)
                        / self.post_len
                    )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))

        # add in exponential part of HBM likelihood
        # this is (negative) the expected number of planets detected by the survey; good sanity check
        norm_constant = 0
        for i in range(self.n_msini_bins):
            for j in range(self.n_e_bins):
                bin_emin = self.ecc_bins[j]
                bin_emax = self.ecc_bins[j + 1]
                norm_constant -= self.completeness[j, 0, i] * (
                    (1 / 3 * As[i] * (bin_emax**3 - bin_emin**3))
                    + (1 / 2 * Bs[i] * (bin_emax**2 - bin_emin**2))
                    + (Cs[i] * (bin_emax - bin_emin))
                )  # TODO: multiply by the width of the msini and a bins here?

        # print(norm_constant)
        log_likelihood += norm_constant

        return log_likelihood

    def sample(self, nsteps, burn_steps=200, nwalkers=100):

        ndim = 3 * self.n_msini_bins  # 3 parabola params for each msini range
        p0 = np.random.uniform(0, 1, size=(nwalkers, ndim))

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
    n_samples = 100  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results

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

    # use bins of e = many (this will only propagate through completeness)
    # and a = 1 (the outermost) and msini = 2
    n_msini_bins = 2
    n_sma_bins = 1
    n_e_bins = 5

    like = HierParab(
        ecc_posteriors,
        msini_posteriors=msini_posteriors,
        sma_posteriors=sma_posteriors,
        n_sma_bins=n_sma_bins,
        n_e_bins=n_e_bins,
        n_msini_bins=n_msini_bins,
    )

    print("Running MCMC!")
    burn_steps = 10  # 500
    nwalkers = 100
    nsteps = 10  # 500

    hbm_samples = like.sample(
        nsteps,
        burn_steps=burn_steps,
        nwalkers=nwalkers,
    )

    savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e_parab"

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    np.savetxt(
        "{}/epop_samples_burn{}_total{}.csv".format(savedir, burn_steps, nsteps),
        hbm_samples,
        delimiter=",",
    )
