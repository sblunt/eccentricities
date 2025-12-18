import numpy as np
import matplotlib.pyplot as plt
import corner
from orbitize.priors import LinearPrior
import glob
import os
import pandas as pd
from radvel.utils import semi_amplitude, initialize_posterior
import scipy
from astropy import units as u, constants as cst
import pickle

"""
NOTE: needs radvel v1.3.8 to load posteriors
"""

# """
# Draws values from a linearly increasing prior, then importance samples
# them to be consistent with a linearly decreasing prior
# """

# n_samples = int(1e5)
# oldLinearPrior = LinearPrior(-0.5, 1.5)
# oldLinearPrior_norm = -0.5 * oldLinearPrior.b**2 / oldLinearPrior.m
# newLinearPrior = LinearPrior(-2, 4.0)
# newLinearPrior_norm = -0.5 * newLinearPrior.b**2 / newLinearPrior.m
# samples = oldLinearPrior.draw_samples(n_samples)

# old_prior_probs = np.exp(oldLinearPrior.compute_lnprob(samples))
# new_prior_probs = np.exp(newLinearPrior.compute_lnprob(samples))
# importance_weights = new_prior_probs / old_prior_probs
# importance_probs = importance_weights / np.sum(importance_weights)

# resampled_samples = np.random.choice(samples, size=int(1e5), p=importance_probs)

# x2plot = np.array([0, 5])
# old_prior_pred = (oldLinearPrior.m * x2plot + oldLinearPrior.b) / oldLinearPrior_norm
# new_prior_pred = (newLinearPrior.m * x2plot + newLinearPrior.b) / newLinearPrior_norm
# plt.figure()
# plt.hist(
#     samples,
#     bins=50,
#     density=True,
#     alpha=0.5,
#     color="purple",
#     label="original samples",
# )
# plt.plot(x2plot, old_prior_pred, color="purple", label="original prior")
# plt.hist(
#     resampled_samples,
#     bins=50,
#     density=True,
#     alpha=0.5,
#     color="grey",
#     label="importance resampled",
# )
# plt.plot(x2plot, new_prior_pred, color="k", label="new prior")
# plt.ylim(0, 1)
# plt.legend()
# plt.savefig("plots/sanity_checks/prior_resampling.png", dpi=250)


# """
# P is drawn from a uniform distribution. This checks the distribution of sma (assuming
# stellar mass is a constant).

# https://math.stackexchange.com/questions/2842360/if-x-has-a-u-1-3-distribution-and-y-x2-find-the-probability-density
# """

# Pmin = 0
# Pmax = 10
# P = np.random.uniform(0, Pmax, int(1e5))
# sma = P ** (2 / 3)

# sma_min = 0
# sma_max = Pmax ** (2 / 3)


# norm_const = (2 / 3 * sma_max ** (3 / 2)) ** (-1)

# plt.figure()
# plt.hist(sma, bins=50, density=True)
# sma2plot = np.linspace(sma_min, sma_max)
# plt.plot(sma2plot, norm_const * sma2plot ** (1 / 2))
# plt.savefig("plots/sanity_checks/P_vs_sma.png", dpi=250)


# """
# Draws values that are uniform in sqrt(e)sin(w)/sqrt(e)cos(w), converts them to e,
# then plots the result to show that it's uniform in e.
# """

# n_samples = int(1e5)
# s2esinw_samples = np.random.uniform(-1, 1, n_samples)
# s2ecosw_samples = np.random.uniform(-1, 1, n_samples)

# e_samples = s2esinw_samples**2 + s2ecosw_samples**2
# omega_samples = np.degrees(np.arctan2(s2ecosw_samples, s2esinw_samples))

# bins = 20

# fig = corner.corner(
#     np.transpose([e_samples, omega_samples]),
#     labels=["ecc", "$\\omega$ [deg]"],
#     bins=bins,
# )


# fig.axes[0].hist(e_samples[e_samples < 1], color="rebeccapurple", bins=bins, alpha=0.5)


# fig.axes[3].hist(
#     omega_samples[e_samples < 1],
#     color="rebeccapurple",
#     alpha=0.5,
#     bins=bins,
# )

# plt.savefig("plots/sanity_checks/sqrt_e_samples.png", dpi=250)

# """
# Checks how uniformly sampling in e translates to sqrt(1-e^2)
# """

# n_samples = int(1e4)
# e = np.random.uniform(0, 1, size=n_samples)

# e2plot = np.linspace(0, 1, 100, endpoint=False)

# plt.figure()
# plt.hist(np.sqrt(1 - e**2), bins=50, density=True)
# plt.plot(e2plot, e2plot / np.sqrt(1 - e2plot**2))
# plt.savefig("plots/sanity_checks/sqrt1minusesq.png", dpi=250)

# """
# Checks how uniformly sampling in P translates to P^(1/3)
# """

# n_samples = int(1e4)
# Pmax = 30
# P = np.random.uniform(0, Pmax, size=n_samples)

# P2plot = np.linspace(0, Pmax ** (1 / 3), 100, endpoint=False)
# norm_const = 3 / Pmax

# plt.figure()
# plt.hist(P ** (1 / 3), bins=50, density=True)
# plt.plot(P2plot, norm_const * P2plot**2)
# plt.savefig("plots/sanity_checks/P13.png", dpi=250)

# """
# Checks how uniformly sampling in P and e translates to P^(1/3)sqrt(1-e^2)
# https://math.stackexchange.com/questions/55684/pdf-of-product-of-variables
# """

# np.random.seed(1)

# n_samples = int(1e5)
# Pmax = 30
# Kmax = 100
# P = np.random.uniform(0, Pmax, size=n_samples)
# e = np.random.uniform(0, 1, size=n_samples)

# z2plot = np.linspace(0, Pmax ** (1 / 3), 100)
# norm_const = 3 / Pmax

# plt.figure()
# plt.hist(np.sqrt(1 - e**2) * P ** (1 / 3), bins=50, density=True)

# plt.plot(
#     z2plot, 3 * z2plot / Pmax ** (2 / 3) * np.sqrt(1 - z2plot**2 * Pmax ** (-2 / 3))
# )
# plt.savefig("plots/sanity_checks/Pe.png", dpi=250)

# """
# Checks how uniformly sampling in P, e, and K translates to P^(1/3)sqrt(1-e^2)K
# and to Msini (which is the same expression as above, with K in multiples of
# 28.4329 m/s (Jupiter around Earth), P in years, and all multiplied by Mtot^(2/3),
# with Mtot in Msol)
# """


# n_samples = int(1e5)
# Pmax = 4  # [units of yr]
# Kmax_ms = 10_000  # [m/s]
# st_mass = 0.7
# Kmax = Kmax_ms / 28.4329 * st_mass ** (2 / 3)
# print(Kmax)
# # Kmax = 100  # [units of Jupiter semiamplitude times (stellar mass)^(2/3) (i.e. 1 means Mst^(2/3) * 28.4329 m/s)]
# P = np.random.uniform(0, Pmax, size=n_samples)  # [yr]
# e = np.random.uniform(0, 1, size=n_samples)  # []
# K = np.random.uniform(0, Kmax, size=n_samples)

# plt.figure()
# plt.hist(K * np.sqrt(1 - e**2) * P ** (1 / 3), bins=50, density=True)

# plotme = np.linspace(0, Pmax ** (1 / 3) * Kmax, int(1e3))
# print(np.max(plotme))


# M = Kmax * Pmax ** (2 / 3)
# A = Pmax ** (-2 / 3)

# print(M)
# print(A)


# def expr(x, A):
#     term1 = 1 / (2 * np.sqrt(A)) * np.arcsin(np.sqrt(A) * x)
#     term2 = x / 2 * np.sqrt(1 - (A * x**2))
#     return term1 + term2


# pdf = 3 / M * (expr(Pmax ** (1 / 3), A) - expr(plotme / Kmax, A))

# print(plotme)

# plt.plot(plotme, pdf)
# plt.savefig("plots/sanity_checks/PeK.png", dpi=250)

# """
# Make a plot of K divided by average RV unc (assumed to be 1 m/s) for all the
# super-Jovian planets in the sample
# """
# semi_amps = []

# origin = "resampled"
# for post_path in glob.glob(f"lee_posteriors/{origin}/ecc_*.csv"):

#     ecc_post = pd.read_csv(post_path).values.flatten()
#     post_len = len(ecc_post)

#     st_name = post_path.split("/")[-1].split("_")[1]
#     pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0]

#     stellar_params = pd.read_csv(
#     "/home/sblunt/CLSI/legacy_tables/stellar_parameters.csv", index_col=0
#     )
#     stellar_props = stellar_params[stellar_params.name == st_name]
#     Mstar = stellar_props.mass_c.values[0]
#     Mstar_err = stellar_props.mass_err_c.values[0]
#     mtot_post = np.random.normal(Mstar, Mstar_err, size=len(ecc_post))

#     msini_post = pd.read_csv(
#         f"lee_posteriors/{origin}/msini_{st_name}_{pl_num}.csv"
#     ).values.flatten()
#     sma_post = pd.read_csv(
#         f"lee_posteriors/{origin}/sma_{st_name}_{pl_num}.csv"
#     ).values.flatten()
#     per_post = np.sqrt(sma_post**3 / mtot_post)

#     if np.median(msini_post)> 300:
#         semi_amps.append(np.min(semi_amplitude(msini_post, per_post, mtot_post, ecc_post, Msini_units='earth',)))

# plt.figure()
# plt.hist(np.array(semi_amps), range=(0,500),bins=25)
# plt.xlabel('K [m/s]')
# plt.savefig('plots/sanity_checks/semiamp.png',dpi=250
# )


"""
Inclination marginalization toy model
"""

# # Assume occurrence rate as a function of mass bins
# msini = 3.5
# mass_bin_lims = np.array([0, 2, 4, 6, 8])
# occurrence = np.array([2, 5, 10, 20])


# # compute the inclination limits that correspond to the boundaries of all larger mass bins
# inc_limits = np.arcsin(msini / mass_bin_lims)

# inc_cutoffs = np.append(np.append(np.pi / 2, inc_limits[inc_limits > 0]), 0)
# cosi_cutoffs = np.cos(inc_cutoffs)

# occurrences_to_draw_from = np.append(occurrence[(mass_bin_lims[1:] > msini)], 0)
# print(occurrences_to_draw_from)

# # draw cosi from a step-uniform distribution
# random_cosis = scipy.stats.rv_histogram((occurrences_to_draw_from, cosi_cutoffs)).rvs(
#     size=100_000
# )


# random_masses = msini / np.sin(np.arccos(random_cosis))

# fig, ax = plt.subplots()
# plt.hist(random_masses, bins=50, color="rebeccapurple")
# plt.xlabel("Mass")
# for i in inc_cutoffs:
#     ax.axvline(msini / np.sin(i), ls="--")
# plt.savefig("plots/sanity_checks/inc_marginalization.png", dpi=250)

"""
2-for-1 plots for planets in highest mass bin

from Wittenmyer et al 2019:

1. The fitted eccentricity must be at least 3sigma from zero.
2. The fitted velocity amplitude K must be at least four times its own uncertainty: K/sigmaK > 4.0. 
3. The fitted velocity amplitude must be at least 1.23 times larger than the rms scatter about the fit: K/rms > 1.23. 
4. The fitted period must be less than 1.5 times the total duration of the observations. 
5. The rms of the fit must be less than three times the mean measurement uncertainty
"""


fig, ax = plt.subplots(1, 5, figsize=(8, 2), sharey=True)
plt.subplots_adjust(wspace=0.1, bottom=0.2)

ecc_sigmas = []
semiamp_sigmas = []
semiamp_rms = []
p_durs = []
measunc_rms = []
semiamps = []

# this table is based on isoclassify, using Gaia DR2 parallaxes and K band magnitudes when known
# (Rosenthal+ Table 2)
stellar_params = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/stellar_parameters.csv", index_col=0
)

for post_path in glob.glob("lee_posteriors/resampled/ecc_*.csv"):

    n_samples = 10_000

    ecc_post = pd.read_csv(post_path).values.flatten()
    post_len = len(ecc_post)

    st_name = post_path.split("/")[-1].split("_")[1]
    pl_num = post_path.split("/")[-1].split("_")[2].split(".")[0].split("pl")[1]

    msini_post = pd.read_csv(
        f"lee_posteriors/resampled/msini_{st_name}_pl{pl_num}.csv"
    ).values.flatten()
    sma_post = pd.read_csv(
        f"lee_posteriors/resampled/sma_{st_name}_pl{pl_num}.csv"
    ).values.flatten()

    if (
        np.median(msini_post) > 1000
        and np.median(sma_post) > 0.10533575
        and np.median(sma_post) < 4.55973325
    ):  # if it's in the highest mass bin

        ecc_sigmas.append(np.median(ecc_post) / np.std(ecc_post))

        stellar_props = stellar_params[stellar_params.name == st_name]
        Mstar = stellar_props.mass_c.values[0]
        Mstar_err = stellar_props.mass_err_c.values[0]
        Mstar_prior = np.random.normal(Mstar, Mstar_err, size=len(ecc_post))

        per_post = (np.sqrt((sma_post**3 / Mstar_prior)) * u.yr).to(u.day).value

        samiamp_post = semi_amplitude(
            msini_post, per_post, Mstar_prior, ecc_post, Msini_units="earth"
        )
        semiamps.append(np.median(samiamp_post))
        semiamp_sigmas.append(np.median(samiamp_post) / np.std(samiamp_post))
        # if np.median(samiamp_post) / np.std(samiamp_post) < 4:
        #     print(
        #         "{} fails with K sigma: {:.2f}".format(
        #             st_name, np.median(samiamp_post) / np.std(samiamp_post)
        #         )
        #     )
        # if np.median(ecc_post) / np.std(ecc_post) < 3:
        #     print(
        #         "{} fails with e sigma:{:.2f}".format(
        #             st_name, np.median(ecc_post) / np.std(ecc_post)
        #         )
        #     )

        # load the MAP fit
        with open(
            "/home/sblunt/eccentricities/lee_posteriors/run_final/{}/post_final.pkl".format(
                st_name
            ),
            "rb",
        ) as f:
            posterior = pickle.load(f)
            rms = np.std(posterior.likelihood.residuals())
            semiamp_rms.append(posterior.params[f"k{pl_num}"].value / rms)
            # if semiamp_rms[-1] < 1.23:
            #     print("{} fails with K/rms:{:.2f}".format(st_name, semiamp_rms[-1]))

            obs_duration = np.max(posterior.likelihood.x) - np.min(
                posterior.likelihood.x
            )
            per = posterior.params[f"per{pl_num}"].value
            p_durs.append(obs_duration / per)
            # if obs_duration / per < 1.5:
            #     print(
            #         "{} fails with observation duration/P: {:.2f}".format(
            #             st_name, obs_duration / per
            #         )
            #     )
            mean_meas_unc = np.median(posterior.likelihood.errorbars())
            measunc_rms.append(mean_meas_unc / rms)
            # if rms / mean_meas_unc < 1 / 3:
            #     print(
            #         "{} fails with rms/mean meas. unc: {:.2f}".format(
            #             st_name, mean_meas_unc / rms
            #         )
            #     )

print(np.median(semiamps))

ax[0].hist(
    ecc_sigmas,
    bins=np.logspace(np.log10(3), np.log10(5000), 40),
    color="grey",
    histtype="stepfilled",
)
ax[0].set_ylabel("# posteriors")
ax[0].set_xlabel("e/$\sigma_{{\\mathrm{{e}}}}$")
ax[0].axvline(3, color="k", ls="--")
ax[0].set_xscale("log")

ax[1].hist(semiamp_sigmas, color="grey", histtype="stepfilled", bins=40)
ax[1].set_xlabel("K/$\sigma_{{\\mathrm{{K}}}}$")
ax[1].axvline(4, color="k", ls="--")

ax[2].hist(
    semiamp_rms,
    color="grey",
    histtype="stepfilled",
    bins=np.logspace(np.log10(1), np.log10(2500), 40),
)
ax[2].set_xlabel("K$_{{\\mathrm{{MAP}}}}$/RMS")
ax[2].axvline(1.23, color="k", ls="--")
ax[2].set_xscale("log")

ax[3].hist(
    p_durs,
    color="grey",
    histtype="stepfilled",
    bins=np.logspace(np.log10(1), np.log10(500), 40),
)
ax[3].set_xlabel("obs dir./P$_{{\\mathrm{{MAP}}}}$")
ax[3].axvline(1.5, color="k", ls="--")
ax[3].set_xscale("log")

ax[4].hist(np.array(measunc_rms), color="grey", histtype="stepfilled", bins=40)
ax[4].set_xlabel("mean meas. unc./RMS")
ax[4].axvline(1 / 3, color="k", ls="--")


for a in ax:
    a.set_box_aspect()
    a.set_ylim(0, 7)
plt.savefig("plots/sanity_checks/2for1.png", dpi=250)


""""
Grether+ 2006 says 11% of solar-type stars have close (<5 yr) stellar companions.
Assume all of these have masses 0.1 Msun (worst case scenario). Assuming they are
uniformly oriented on surface of sphere, then we expect the number in the
msini range to be what's calculated below.
"""

gamma = 0.008

n_stars_cps = 719

mtrue = (13 * u.M_jup).to(u.M_sun).value  # [Msun]

msini_bin_limits = np.array(
    [0.00180209, 0.01802094]
)  # [Msun] msini bin of interest (where ecc peak occurs)

mratio_hi = msini_bin_limits[1] / mtrue
mratio_lo = msini_bin_limits[0] / mtrue

if mratio_hi > 1:
    mratio_hi = 1

n_interloping_binaries = (
    gamma * n_stars_cps * (np.cos(np.arcsin(mratio_lo)) - np.cos(np.arcsin(mratio_hi)))
)

# print(n_interloping_binaries)
