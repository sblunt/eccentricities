import pandas as pd
from radvel.basis import Basis
from radvel.utils import Msini, semi_major_axis
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u, constants as cst
import scipy

"""
Grabs ecc, msini, and sma posteriors for the planet sample and writes them as csvs 
to be ingested into epop!
"""

legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0, comment="#"
)

# this table is based on isoclassify, using Gaia DR2 parallaxes and K band magnitudes when known
# (Rosenthal+ Table 2)
stellar_params = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/stellar_parameters.csv", index_col=0
)


def compute_importance_probabilities(
    sma_prior,
    msini_prior,
    sma_limits,
    msini_limits,
    sma_posterior,
    msini_posterior,
    loguniform=False,
    saveplot=True,
):

    # importance sample the msini prior & posterior down to a uniform prior on log(msini) (which is same as applying 1/x prior)
    # https://stats.stackexchange.com/questions/493868/using-importance-sampling-for-prior-sensitivity-analysis-in-bayesian-modeling

    msini_min, msini_max = (
        msini_limits  # min & values of the effective prior we're setting
    )
    sma_min, sma_max = sma_limits

    good_prior_indices = (
        (msini_prior > msini_min)
        & (msini_prior < msini_max)
        & (sma_prior > sma_min)
        & (sma_prior < sma_max)
    )

    msini_prior_inrange = msini_prior[good_prior_indices]
    sma_prior_inrange = sma_prior[good_prior_indices]

    # to estimate the value of the actual prior applied, take a histogram of msini_prior values in the
    # region of the posterior, then use the heights of the posterior as the probabilities
    prior_probs, msini_bins, sma_bins = np.histogram2d(
        msini_prior_inrange, sma_prior_inrange, density=True
    )

    # plot this effective prior
    if saveplot:
        fig, ax = plt.subplots(3, 1)

        ax[0].pcolormesh(msini_bins, sma_bins, prior_probs)
        ax[0].set_xlabel("Msini")
        ax[0].set_ylabel("sma [au]")

        msini_min = 0
        msini_max = 1000
        A = 3 * (msini_max ** (3) - msini_min ** (3)) ** (-1)

        ax[1].hist(
            msini_prior[msini_prior > 0],
            bins=50,
            density=True,
            range=(msini_min, msini_max),
        )  # _inrange)
        msini2plot = np.linspace(msini_min, msini_max, int(1e3))
        ax[1].plot(msini2plot, A * msini2plot ** (2))

        sma_max = 3
        sma_min = 0
        ax[2].hist(
            sma_prior, bins=50, density=True, range=(sma_min, sma_max)
        )  # _inrange)

        sma2plot = np.linspace(sma_min, sma_max, int(1e3))
        A = 3 / 2 * (sma_max ** (3 / 2) - sma_min ** (3 / 2)) ** (-1)
        ax[2].plot(sma2plot, A * sma2plot ** (1 / 2))
        plt.savefig(
            "/home/sblunt/eccentricities/lee_posteriors/resampled/effective_prior{}_pl{}.png".format(
                pl[1].hostname, int(pl[1].pl_index)
            ),
            dpi=250,
        )

    # interpolate this effective prior to get effective prior probabilities of the posterior samples

    probabilityInterpolater = scipy.interpolate.RegularGridInterpolator(
        (msini_bins[:-1], sma_bins[:-1]),
        prior_probs,
        bounds_error=False,
        fill_value=0.0,
    )

    posterior_pts = np.array([msini_posterior, sma_posterior])
    old_prior_probs_interpolated = probabilityInterpolater(posterior_pts.T)

    if not loguniform:
        A_msini = 1 / (msini_max - msini_min)
        msini_sample_probs = A_msini
        A_sma = 1 / (sma_max - sma_min)
        sma_sample_probs = A_sma

        new_prior_probs = msini_sample_probs * sma_sample_probs

    importance_weights = new_prior_probs / old_prior_probs_interpolated
    importance_weights[~np.isfinite(importance_weights)] = 0.0
    importance_probs = importance_weights / np.sum(importance_weights)

    return importance_probs


# A_msini = np.log(msini_max) - np.log(
#     msini_min
# )  # integration constant to make the prior proper
#

# random_samples = np.random.uniform(0, 1, size=len(mass_posterior))
# msini_sample_probs = A_msini  # / msini_posterior

# msini_importance_weights = (
#     msini_sample_probs / msini_prior_probs_interpolated
# ) / np.sum(msini_sample_probs / msini_prior_probs_interpolated)

# msini_resampled_prior = np.random.choice(
#     msini_prior, size=int(1e4), p=msini_importance_weights
# )

# rejection sample the sma prior & posterior down to a uniform prior on log(sma)
# sma_min = np.min(sma_posterior)  # min & values of the effective prior we're setting
# sma_max = np.max(sma_posterior)
# A = 1 / (sma_max - sma_min)  # np.log(sma_max) - np.log(
# #     sma_min
# # )  # integration constant to make the prior proper

# sma_sample_probs = A  # / sma_posterior


# construct the eccentricity posterior for each
for pl in legacy_planets.iterrows():
    if pl[1].status not in ["A", "R", "N"]:
        print("{} pl {}".format(pl[1].hostname, int(pl[1].pl_index)))
        print("Copying number {}".format(pl[0]))
        starname = pl[1].hostname
        if (
            starname == "112914"  # != "213472" # TODO: fix
        ):  # this one was modeled with thejoker (as was 26161, which doesn't seem to be in the results) (I'm not interested in partial orbits here)

            plnum = int(pl[1].pl_index)
            chains = pd.read_csv(
                "/home/sblunt/eccentricities/lee_posteriors/run_final/{}/chains.csv.tar.bz2".format(
                    starname
                ),
                compression="bz2",
            )
            with open(
                "/home/sblunt/eccentricities/lee_posteriors/run_final/{}/post_final.pkl".format(
                    starname
                ),
                "rb",
            ) as f:
                posterior = pickle.load(f)
            basis_name = posterior.likelihood.params.basis.name
            # we're making the assumption that all of these posteriors were computed with this
            # basis, so raise an error if that assumption is violated.
            assert "secosw sesinw" in basis_name
            n_planets = posterior.likelihood.params.num_planets
            myBasis = Basis(basis_name, n_planets)
            df_synth = myBasis.to_synth(chains)

            ecc_posterior = df_synth["e{}".format(plnum)].values

            stellar_props = stellar_params[stellar_params.name == starname]
            Mstar = stellar_props.mass_c.values[0]
            Mstar_err = stellar_props.mass_err_c.values[0]
            Mstar_prior = np.random.normal(Mstar, Mstar_err, size=len(ecc_posterior))

            msini_posterior = Msini(
                df_synth["k{}".format(plnum)].values,
                df_synth["per{}".format(plnum)].values,
                Mstar_prior,
                df_synth["e{}".format(plnum)].values,
                Msini_units="jupiter",
            )

            cosi = np.random.uniform(-1, 1, size=len(ecc_posterior))
            inc = np.arccos(cosi)
            mass_posterior = msini_posterior / np.sin(inc)
            sma_posterior = semi_major_axis(
                df_synth["per{}".format(plnum)].values, Mstar_prior
            )

            # construct the effective priors on sma and msini
            Pmin = np.min(df_synth["per{}".format(plnum)].values)
            Pmax = np.max(df_synth["per{}".format(plnum)].values)
            period_prior = np.random.uniform(
                0,
                10 * Pmax,
                size=len(mass_posterior),
            )
            sma_prior = semi_major_axis(period_prior, Mstar_prior)
            Kmin = np.min(df_synth["k{}".format(plnum)].values)
            Kmax = np.max(df_synth["k{}".format(plnum)].values)
            K_prior = np.random.uniform(
                0,
                10 * Kmax,
                size=len(mass_posterior),
            )
            e_prior = np.random.uniform(0, 1, size=len(mass_posterior))
            msini_prior = Msini(
                K_prior, period_prior, Mstar_prior, e_prior, Msini_units="jupiter"
            )

            importance_probs = compute_importance_probabilities(
                sma_prior,
                msini_prior,
                (np.min(sma_posterior), np.max(sma_posterior)),
                (np.min(msini_posterior), np.max(msini_posterior)),
                sma_posterior,
                msini_posterior,
            )

            # use importance weights to resample whole 3d posterior
            resampled_indices = np.random.choice(
                np.arange(len(sma_posterior)), size=int(1e3), p=importance_probs
            )

            sma_resampled_posterior = sma_posterior[resampled_indices]
            ecc_resampled_posterior = ecc_posterior[resampled_indices]
            msini_resampled_posterior = msini_posterior[resampled_indices]

            fig, ax = plt.subplots(3, 1)
            ax[0].hist(
                sma_resampled_posterior,
                range=(np.min(sma_posterior), np.max(sma_posterior)),
                bins=50,
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                color="purple",
                label=f"resampled posterior ({len(sma_resampled_posterior)} samples)",
            )
            ax[0].hist(
                sma_posterior,
                bins=50,
                density=True,
                histtype="step",
                alpha=0.2,
                color="k",
                label=f"original posterior ({len(mass_posterior)} samples)",
            )

            # ax[1].hist(
            #     msini_prior,
            #     density=True,
            #     bins=50,
            #     color="purple",
            #     alpha=0.2,
            #     label="prior",
            # )
            ax[1].hist(
                msini_posterior,
                bins=50,
                density=True,
                histtype="step",
                alpha=0.2,
                color="k",
                label="posterior",
            )
            ax[1].hist(
                msini_resampled_posterior,
                range=(np.min(msini_posterior), np.max(msini_posterior)),
                bins=50,
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                color="purple",
                label="resampled posterior",
            )

            ax[2].hist(
                ecc_resampled_posterior,
                range=(np.min(ecc_posterior), np.max(ecc_posterior)),
                bins=50,
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                color="purple",
                label="resampled posterior",
            )
            ax[2].hist(
                ecc_posterior,
                bins=50,
                density=True,
                histtype="step",
                alpha=0.2,
                color="k",
                label="posterior",
            )

            ax[0].legend()
            ax[0].set_xlabel("sma [au]")
            ax[1].set_xlabel("msini [M$_{{\\mathrm{{Jup}}}}$]")
            ax[2].set_xlabel("ecc")
            # ax[0].set_yscale("log")
            # ax[1].set_yscale("log")
            plt.tight_layout()
            plt.savefig(
                "/home/sblunt/eccentricities/lee_posteriors/resampled/priors_{}_pl{}.png".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                dpi=250,
            )
            plt.close()

            # np.savetxt(
            #     "/home/sblunt/eccentricities/lee_posteriors/{}/ecc_{}_pl{}.csv".format(
            #         savedir, pl[1].hostname, int(pl[1].pl_index)
            #     ),
            #     ecc_posterior,
            #     delimiter=",",
            # )
            # np.savetxt(
            #     "/home/sblunt/eccentricities/lee_posteriors/{}/msini_{}_pl{}.csv".format(
            #         savedir, pl[1].hostname, int(pl[1].pl_index)
            #     ),
            #     msini_posterior,
            #     delimiter=",",
            # )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/resampled/msiniRESAMPLED_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                msini_resampled_posterior,
                delimiter=",",
            )
            # np.savetxt(
            #     "/home/sblunt/eccentricities/lee_posteriors/{}/sma_{}_pl{}.csv".format(
            #         savedir, pl[1].hostname, int(pl[1].pl_index)
            #     ),
            #     sma_posterior,
            #     delimiter=",",
            # )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/resampled/smaRESAMPLED_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_resampled_posterior,
                delimiter=",",
            )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/resampled/ecc_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                ecc_resampled_posterior,
                delimiter=",",
            )
