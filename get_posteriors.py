import pandas as pd
from radvel.basis import Basis
from radvel.utils import Msini, semi_major_axis
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u, constants as cst
import scipy

"""
Grabs ecc, msini, and sma posteriors for the planet sample, uses importance
resampling to obtain samples the posteriors assuming they were sampled under
unifom priors on log(sma) and log(msini), and writes them as csvs to be injested 
into the HBM model. This code takes a while because it has to load all the MCMC chains.

NOTE: I'm also including detected stellar binaries here

NOTE: needs radvel v1.3.8 to load posteriors
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
    Pmax,
    Mst_median,
    sma_posterior,
    msini_posterior,
    savename=None,
):
    """
    importance sample the msini prior & posterior down to a uniform prior on log(msini) (which is same as applying 1/x prior)
    https://stats.stackexchange.com/questions/493868/using-importance-sampling-for-prior-sensitivity-analysis-in-bayesian-modeling

    Pmax: [days]
    Kmax: [m/s]
    """

    Pmax_yr = Pmax / 365.25
    msini_max = np.max([np.max(msini_prior), 2 * np.max(msini_posterior)])
    Kmax_unitless = msini_max / Pmax_yr ** (1 / 3)

    # some useful constants
    sma_max = ((Pmax_yr) ** 2 * Mst_median) ** (1 / 3)

    # compute old sma prior probs analytically
    def calc_sma_prior(x):

        norm_const = (2 / 3 * sma_max ** (3 / 2)) ** (-1)
        return norm_const * x ** (1 / 2)

    def expr(x):
        A = Pmax_yr ** (-2 / 3)
        term1 = 1 / (2 * np.sqrt(A)) * np.arcsin(np.sqrt(A) * x)
        term2 = x / 2 * np.sqrt(1 - (A * x**2))
        return term1 + term2

    # compute old msini prior probs analytically
    def calc_msini_prior(msini, msini2plot):

        M = Kmax_unitless * Pmax_yr ** (2 / 3)

        # NOTE: I replaced the commented out portion (the analytical answer) with the nanmax
        # because sometimes it was failing. Seems like a fine approximation.
        return (
            3
            / M
            * (
                -expr(msini / Kmax_unitless)
                + np.nanmax(expr(msini2plot / Kmax_unitless))
            )
        )  # expr(msini_max / Kmax_unitless)

    # overplot old prior probs computed numerically and analytical answer (to check analytical math)
    if savename is not None:
        _, ax = plt.subplots(2, 1)
        ax[0].hist(
            sma_prior,
            bins=50,
            density=True,
            color="rebeccapurple",
            alpha=0.5,
            label="numerically calculated",
        )
        sma2plot = np.linspace(0, sma_max, int(1e3))
        ax[0].plot(
            sma2plot,
            calc_sma_prior(sma2plot),
            color="k",
            label="analytically calculated",
        )
        ax[0].legend()
        ax[0].set_xlabel("$a$ [au]")

        ax[1].hist(
            msini_prior[msini_prior > 0],
            bins=50,
            density=True,
            color="rebeccapurple",
            alpha=0.5,
        )
        msini2plot = np.linspace(0, msini_max, int(1e3))

        # TODO: something is a little off with the math of this prior. Edge effects?
        # Doesn't seem to make any impact whatsoever for this paper though.
        msini_prior_vals = calc_msini_prior(msini2plot, msini2plot)
        ax[1].plot(msini2plot, msini_prior_vals, color="k")
        ax[1].set_xlabel("M$\sin{i}$ [M$_{\\oplus}$]")

        for a in ax:
            a.set_ylabel("probability")
        plt.tight_layout()
        plt.savefig(savename, dpi=250)

    # compute old prior prob (probability of the two multiplied)
    log_old_prior_probs = np.log(
        calc_msini_prior(msini_posterior, msini2plot)
    ) + np.log(calc_sma_prior(sma_posterior))

    # compute new sma prior prob (uniform in logspace = 1/x prior pdf)
    sma_min = np.min(sma_posterior)
    sma_norm = 1 / (np.log(sma_max) - np.log(sma_min))
    sma_new_prior_prob = sma_norm / sma_posterior

    # compute new msini prior prob (loguniform)
    msini_min = np.min(msini_posterior[msini_posterior > 0])
    msini_norm = 1 / (np.log(msini_max) - np.log(msini_min))
    msini_new_prior_prob = msini_norm / msini_posterior
    log_new_prior_probs = np.log(sma_new_prior_prob) + np.log(msini_new_prior_prob)

    # compute total new prior prob by multiplying these two
    log_importance_weights = log_new_prior_probs - log_old_prior_probs
    importance_weights = np.exp(log_importance_weights)
    importance_weights[~np.isfinite(importance_weights)] = 0.0
    importance_weights[np.isnan(importance_weights)] = 0.0
    importance_probs = importance_weights / np.nansum(importance_weights)

    return importance_probs


# construct the eccentricity posterior for each
for pl in legacy_planets.iterrows():
    if pl[1].status not in ["A", "R", "N"]:
        print("{} pl {}".format(pl[1].hostname, int(pl[1].pl_index)))
        print("Copying number {}".format(pl[0]))
        starname = pl[1].hostname
        if starname not in [
            "141399"  # ,"111031"
        ]:  # (can use this logic to rerun resampling for only a subset of objects)
            continue
        if (
            starname != "213472"
        ):  # this one was modeled with thejoker (as was 26161, which doesn't seem to be in the results) (I'm not interested in partial orbits here)

            plnum = int(pl[1].pl_index)

            chains = pd.read_csv(
                "ee_posteriors/run_final/{}/chains.csv.tar.bz2".format(starname),
                compression="bz2",
            )
            with open(
                "lee_posteriors/run_final/{}/post_final.pkl".format(starname),
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
            per_posterior = df_synth["per{}".format(plnum)].values
            k_posterior = df_synth["k{}".format(plnum)].values

            stellar_props = stellar_params[stellar_params.name == starname]
            Mstar = stellar_props.mass_c.values[0]
            Mstar_err = stellar_props.mass_err_c.values[0]
            Mstar_prior = np.random.normal(Mstar, Mstar_err, size=len(ecc_posterior))

            msini_posterior = Msini(
                k_posterior,
                per_posterior,
                Mstar_prior,
                ecc_posterior,
                Msini_units="earth",
            )
            good_indices = np.where(msini_posterior > 0)[0]

            ecc_posterior = ecc_posterior[good_indices]
            per_posterior = per_posterior[good_indices]
            Mstar_prior = Mstar_prior[good_indices]
            k_posterior = k_posterior[good_indices]
            msini_posterior = msini_posterior[good_indices]

            cosi = np.random.uniform(-1, 1, size=len(ecc_posterior))
            inc = np.arccos(cosi)
            mass_posterior = msini_posterior / np.sin(inc)
            sma_posterior = semi_major_axis(per_posterior, Mstar_prior)

            # construct the effective priors on sma and msini
            Pmax = 2 * np.max(per_posterior)
            period_prior = np.random.uniform(
                0,
                Pmax,
                size=len(mass_posterior),
            )
            sma_prior = semi_major_axis(period_prior, Mstar_prior)
            Kmax = 2 * np.max(per_posterior)
            K_prior = np.random.uniform(
                0,
                Kmax,
                size=len(mass_posterior),
            )
            e_prior = np.random.uniform(0, 1, size=len(mass_posterior))
            msini_prior = Msini(
                K_prior, period_prior, Mstar_prior, e_prior, Msini_units="earth"
            )

            importance_probs = compute_importance_probabilities(
                sma_prior,
                msini_prior,
                Pmax,
                Mstar,
                sma_posterior,
                msini_posterior,
                savename="lee_posteriors/resampled/prior_approx_{}_pl{}.png".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
            )

            if np.sum(np.isnan(importance_probs)) > 0:
                print(
                    f"warning: importance probabilities contained {np.sum(np.isnan(importance_probs))} nans"
                )
            importance_probs[np.isnan(importance_probs)] = 0

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
            ax[0].set_xlabel("$a$ [au]")
            ax[1].set_xlabel("M$\sin{{i}}$ [M$_{{\\oplus}}$]")
            ax[2].set_xlabel("$e$")
            ax[1].set_ylabel("relative prob.")

            plt.tight_layout()
            plt.savefig(
                "lee_posteriors/resampled/priors_{}_pl{}.png".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                dpi=250,
            )
            plt.close()

            np.savetxt(
                "lee_posteriors/resampled/msini_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                msini_resampled_posterior,
                delimiter=",",
            )

            np.savetxt(
                "lee_posteriors/resampled/sma_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_resampled_posterior,
                delimiter=",",
            )
            np.savetxt(
                "lee_posteriors/resampled/ecc_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                ecc_resampled_posterior,
                delimiter=",",
            )

            np.savetxt(
                "lee_posteriors/original/msini_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                msini_posterior,
                delimiter=",",
            )

            np.savetxt(
                "lee_posteriors/original/sma_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_posterior,
                delimiter=",",
            )
            np.savetxt(
                "lee_posteriors/original/ecc_{}_pl{}.csv".format(
                    pl[1].hostname, int(pl[1].pl_index)
                ),
                ecc_posterior,
                delimiter=",",
            )
