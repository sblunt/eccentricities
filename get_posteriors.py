import pandas as pd
from radvel.basis import Basis
from radvel.utils import Msini, semi_major_axis
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Grabs ecc, msini, and sma posteriors for the planet sample and writes them as csvs 
to be ingested into epop!
"""

legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0
)
legacy_planets_inseprange = legacy_planets[
    (legacy_planets.axis_med > 0.02) & (legacy_planets.axis_med < 6)
]
legacy_planets = legacy_planets_inseprange[(legacy_planets.mass_med < 15)]

stellar_params = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/stellar_parameters.csv", index_col=0
)

fig, ax = plt.subplots(2, 1, figsize=(15, 5))

# construct the eccentricity posterior for each
for pl in legacy_planets.iterrows():
    if pl[1].status not in ["A", "R", "N"]:
        print("{} pl {}".format(pl[1].hostname, int(pl[1].pl_index)))
        starname = pl[1].hostname
        if starname != "213472":  # this one seems to have very badly converged chains

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
            Mstar_post = np.random.normal(Mstar, Mstar_err, size=len(ecc_posterior))

            msini_posterior = Msini(
                df_synth["k{}".format(plnum)].values,
                df_synth["per{}".format(plnum)].values,
                Mstar_post,
                df_synth["e{}".format(plnum)].values,
                Msini_units="jupiter",
            )

            cosi = np.random.uniform(-1, 1, size=len(ecc_posterior))
            inc = np.arccos(cosi)
            mass_posterior = msini_posterior / np.sin(inc)

            sma_posterior = semi_major_axis(
                df_synth["per{}".format(plnum)].values, Mstar_post
            )

            rand_samples = np.random.choice(len(sma_posterior), size=1_000)

            if pl[1].mass < 1:
                savedir = "lowmass"
                ax[0].hist(
                    msini_posterior[rand_samples],
                    bins=50,
                    # density=True,
                    histtype="step",
                    alpha=0.2,
                    color="purple",
                )
                # ax[0].hist(
                #     mass_posterior,
                #     bins=50,
                #     density=True,
                #     alpha=0.2,
                #     color="purple",
                #     range=(0, 16),
                # )
            elif pl[1].mass >= 1:
                savedir = "highmass"
                ax[1].hist(
                    msini_posterior[rand_samples],
                    bins=50,
                    # density=True,
                    histtype="step",
                    alpha=0.2,
                    color="purple",
                )
                # ax[1].hist(
                #     mass_posterior,
                #     bins=50,
                #     density=True,
                #     alpha=0.2,
                #     color="purple",
                #     range=(0, 16),
                # )

            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/{}/ecc_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                ecc_posterior,
                delimiter=",",
            )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/{}/msini_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                msini_posterior,
                delimiter=",",
            )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/{}/sma_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_posterior,
                delimiter=",",
            )
for a in ax:
    a.set_xlabel("Msini [M$_{{\\mathrm{{J}}}}$]")
    a.set_ylabel("rel. prob.")
ax[0].set_title("Msini < 1 M$_{{\\mathrm{{J}}}}$")
ax[1].set_title("1 M$_{{\\mathrm{{J}}}}$ < Msini < 15 M$_{{\\mathrm{{J}}}}$")
plt.tight_layout()
plt.savefig("plots/msini_sample.png", dpi=250)
