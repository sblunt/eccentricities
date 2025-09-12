import numpy as np
import glob
import pandas as pd
import os
import emcee

# WIP implementation of the proper incliantion marginalization
# Abandoned when it was almost debugged due to reasons enumerated
# in the paper.


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

        # read in 3D completeness model
        self.completeness = np.load(
            "completeness_model/{}{}{}completeness.npy".format(
                n_msini_bins, n_e_bins, n_sma_bins
            )
        )
        self.ecc_bins = np.load(
            "completeness_model/{}ecc_bins.npy".format(n_e_bins)
        )  # TODO: change var name to ecc_bin bounds and same for others
        sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
        msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_msini_bins))
        self.msini_bins = msini_bins

        self.mass_bins = np.append(
            0.1, msini_bins
        )  # these are the bins we will use for the model fit. note that we also fit

        # NOTE: here is where we define the bins as uniformly spaced in log(msini) and log(a),
        # and this propagates to the units of our histogram heights
        # self.msini_bin_widths = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
        self.sma_bin_widths = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])
        self.ecc_bin_widths = self.ecc_bins[1:] - self.ecc_bins[:-1]

        self.n_e_bins = len(self.ecc_bins) - 1
        self.n_sma_bins = len(sma_bins) - 1
        self.n_msini_bins = len(msini_bins) - 1
        self.n_mass_bins = len(self.mass_bins) - 1

        self.n_posteriors = len(self.msini_posteriors)

        # in theory the posteriors could have different lengths, but I downsample them to all have
        # the same length in pre-processing
        self.post_len = len(self.sma_posteriors[0])
        self.completeness_labels = np.nan * np.ones(
            (self.post_len, 3, self.n_posteriors), dtype=int
        )
        # self.mass_labels = np.nan * np.ones(
        #     (self.post_len, self.n_posteriors), dtype=int
        # )

        self.cosi_limits = np.nan * np.ones(
            (self.post_len, self.n_posteriors, self.n_mass_bins + 1),
            dtype=int,
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

                ### THIS ASSUMES UNIFORM COSI IN ALL BINS
                # cosi_samples = np.random.uniform(
                #     -1, 1, size=len(self.msini_posteriors[k])
                # )
                # mass_posterior = self.msini_posteriors[k] / (
                #     np.sin(np.arccos(cosi_samples))
                # )
                # self.mass_posteriors.append(mass_posterior)

                # mass_mask = (mass_posterior >= msini_bins[i]) & (
                #     mass_posterior < msini_bins[i + 1]
                # )
                # self.mass_labels[mass_mask, k] = i
                ###

            for i in range(len(self.mass_bins)):
                ### THIS ASSUMES UNIFORM COSI ONLY IN SINGLE MASS BIN
                # compute the inclination limits that correspond to the boundaries of all larger mass bins
                inc_limits = np.arcsin(self.msini_posteriors[k] / self.mass_bins[i])

                cosi_limits_i = np.cos(inc_limits)

                self.cosi_limits[:, k, i] = cosi_limits_i

            for j in np.arange(self.post_len):
                cosi_limits_i = self.cosi_limits[j, k, :]
                if np.isnan(self.cosi_limits[j, k, :]).any():
                    cosi_limits_i[np.max(np.where(np.isnan(cosi_limits_i))[0])] = 0
                self.cosi_limits[j, k, :] = cosi_limits_i

            ###

        # self.bin_widths = np.zeros((self.n_e_bins, self.n_sma_bins, self.n_msini_bins))
        # for i in range(self.n_e_bins):
        #     for j in range(self.n_sma_bins):
        #         for k in range(self.n_msini_bins):e
        #             self.bin_widths[i, j, k] = (
        #                 self.ecc_bin_widths[i]
        #                 * self.sma_bin_widths[j]
        #                 * self.msini_bin_widths[k]
        #             )

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

                    cosi_limits_i = self.cosi_limits[j, i, :]

                    # cosi_limits_i = np.append(
                    #     np.append(
                    #         self.cosi_limits[j, i, :][self.cosi_limits[j, i, :] > 0]
                    #     ),
                    #     1,  # self.cosi_limits has shape (post_len, n_posteriors, n_msini_bins+1)
                    # )

                    msini_idx = int(msini_idx)
                    msini_value = self.msini_posteriors[i][j]
                    for mass_idx in np.arange(self.n_mass_bins):
                        if msini_value <= self.mass_bins[mass_idx]:

                            # TODO: check that this is approx consistent with easier way of doing it
                            # TODO: keep both implementations & check them against each other
                            system_sums[i] += (
                                self.completeness[ecc_idx, sma_idx, msini_idx]
                                * 2
                                * (
                                    histogram_heights[
                                        ecc_idx, sma_idx, mass_idx - 1
                                    ]  # basically we're weighting each histogram height (occurrence rate in mass space) by the fraction of inclination probability space
                                    * (
                                        cosi_limits_i[mass_idx]
                                        - cosi_limits_i[mass_idx - 1]
                                    )
                                )
                                / self.post_len
                            )

        log_likelihood = np.sum(np.nan_to_num(np.log(system_sums), neginf=0.0))
        # print(log_likelihood)

        # add in exponential part of HBM likelihood
        # this is (negative) the expected number of planets detected by the survey; good sanity check
        ## OLD:
        # norm_constant = -np.sum(self.completeness * histogram_heights * self.bin_widths)
        #######
        def _integral(x, B):
            # copied this from wolfram. god bless symbolic integrators.
            # indefinite integral of 1 - arcsin(B/y)dy evaluated at y=x
            xpr = x / B
            integrated_value = (
                x
                - (
                    xpr * np.arcsin(1 / xpr)
                    + np.log(np.abs(xpr + np.tan(np.arccos(1 / xpr))))
                )
                * B
            )
            return integrated_value

        # TODO: try caching this for speed boost, but be careful of state
        def _integral_under_msini(msini):
            # this is the integral over the area that looks like a rectangle with bite out of top right corner
            integral = 0
            for mass_idx in np.arange(self.n_mass_bins):
                theta_n = histogram_heights[
                    ecc_idx, sma_idx, mass_idx
                ]  # fitted paramter

                inc_intercept_up = np.arcsin(msini / self.mass_bins[mass_idx + 1])
                inc_intercept_down = np.arcsin(msini / self.mass_bins[mass_idx])

                if np.isnan(inc_intercept_up) and np.isnan(
                    inc_intercept_down
                ):  # integral is over square area
                    integral += (
                        2
                        * theta_n
                        * (self.mass_bins[mass_idx + 1] - self.mass_bins[mass_idx])
                    )

                elif not np.isnan(inc_intercept_up) and not np.isnan(
                    inc_intercept_down
                ):  # integral is bounded by msini = const curve
                    integral += (
                        2
                        * theta_n
                        * (
                            _integral(self.mass_bins[mass_idx + 1], msini)
                            - _integral(self.mass_bins[mass_idx], msini)
                        )
                    )

                else:
                    assert not np.isnan(inc_intercept_up) and np.isnan(
                        inc_intercept_down
                    )
                    # account for case of round+square
                    integral += (
                        2
                        * theta_n
                        * (
                            _integral(self.mass_bins[mass_idx + 1], msini)
                            - _integral(msini, msini)
                        )
                    )
                    integral += 2 * theta_n * (msini - self.mass_bins[mass_idx])

            import pdb

            pdb.set_trace()
            return integral

        norm_constant = 0
        # integrate mass and inc over msini boundaries, multiplying at each step by completeness and a/e integral
        for ecc_idx in np.arange(self.n_e_bins):
            for sma_idx in np.arange(
                self.n_sma_bins
            ):  # TODO: can I vectorize the ecc/sma multiplication for speed boost?
                for msini_bin_idx in np.arange(self.n_msini_bins)[1:]:

                    print(self.msini_bins[msini_bin_idx])
                    Q_n = self.completeness[ecc_idx, sma_idx, msini_bin_idx]
                    msini_integral = _integral_under_msini(
                        self.msini_bins[msini_bin_idx]
                    ) - _integral_under_msini(
                        self.msini_bins[msini_bin_idx - 1]
                    )  # TODO: why is this negative?

                    norm_constant += (
                        -Q_n
                        * msini_integral
                        * self.ecc_bin_widths[ecc_idx]
                        * self.sma_bin_widths[sma_idx]
                    )
        print(norm_constant)
        log_likelihood += norm_constant

        return log_likelihood

    def sample(self, nsteps, burn_steps=200, nwalkers=100):

        ndim = self.n_e_bins * self.n_sma_bins * self.n_mass_bins
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
    n_samples = 50  # 999  # according to Hogg paper, you can go as low as 50 samples per posterior and get reasonable results

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

    n_msini_bins = 4
    n_sma_bins = 2
    n_e_bins = 4

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
    nsteps = 1  # 500

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

    ### Original (incorrect) formalism for inc marginalization, keeping here for now:
    # THIS ASSUMES UNIFORM COSI ACROSS ALL BINS
    # mass_idx = self.mass_labels[j, i]
    ###

    # ### THIS ASSUMES UNIFORM COSI PER MASS BIN
    # msini = self.msini_posteriors[i][j]

    # if not np.isnan(ecc_idx + sma_idx) and msini > self.msini_bins[0]:

    #     ecc_idx = int(ecc_idx)
    #     sma_idx = int(sma_idx)

    #     # we're assuming here occurrence in bin greater than highest mass bin is 0
    #     occurrences_to_draw_from = np.append(
    #         histogram_heights[
    #             ecc_idx, sma_idx, self.msini_bins[1:] > msini
    #         ],
    #         0,
    #     )

    #     cosi_limits_i = np.append(
    #         np.append(
    #             0, self.cosi_limits[j, i, :][self.cosi_limits[j, i, :] > 0]
    #         ),
    #         1,  # cosi_limits has shape (post_len, n_posteriors, n_msini_bins+1)
    #     )

    #     # draw cosi from a random step-uniform distribution
    #     random_cosis = cached_rvhist(
    #         occurrences_to_draw_from,
    #         cosi_limits_i,
    #     )

    #     random_mass = msini / np.sin(np.arccos(random_cosis))

    #     mass_idx = np.where(
    #         (random_mass < self.msini_bins[1:])
    #         & (random_mass > self.msini_bins[:-1])
    #     )[0]

    #     ###
