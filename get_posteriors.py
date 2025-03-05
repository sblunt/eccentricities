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
    (legacy_planets.axis_med > 0.1) & (legacy_planets.axis_med < 6)
]
legacy_planets = legacy_planets_inseprange[(legacy_planets.mass_med < 15)]

# this table is based on isoclassify, using Gaia DR2 parallaxes and K band magnitudes when known
# (Rosenthal+ Table 2)
stellar_params = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/stellar_parameters.csv", index_col=0
)
# construct the eccentricity posterior for each
for pl in legacy_planets.iterrows():
    if pl[1].status not in ["A", "R", "N"]:
        print("{} pl {}".format(pl[1].hostname, int(pl[1].pl_index)))
        starname = pl[1].hostname
        if (
            starname != "213472"
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

            # construct the effective priors on sma and msini
            period_prior = np.random.uniform(
                0,
                365
                * 8
                ** (
                    3 / 2
                ),  # corresponds to 8 au, which is larger than the sample I'm defining (want use common prior transform for all objects)
                size=len(mass_posterior),
            )
            sma_prior = semi_major_axis(period_prior, Mstar_post)
            K_prior = np.random.uniform(
                0,
                1500,  # corresponds to ~msini=15 Mjup around 1Msol at 0.1 au, which is larger than the sample I'm defining (want use common prior transform for all objects)
                size=len(mass_posterior),
            )
            e_prior = np.random.uniform(0, 1, size=len(mass_posterior))
            msini_prior = Msini(
                K_prior, period_prior, Mstar_post, e_prior, Msini_units="jupiter"
            )

            sma_posterior = semi_major_axis(
                df_synth["per{}".format(plnum)].values, Mstar_post
            )

            if pl[1].mass < 1:
                savedir = "lowmass"
            elif pl[1].mass >= 1:
                savedir = "highmass"

            fig, ax = plt.subplots(2, 1)
            ax[0].hist(
                sma_prior,
                density=True,
                bins=50,
                color="purple",
                alpha=0.2,
                label="prior",
            )
            ax[0].hist(
                sma_posterior,
                bins=50,
                density=True,
                histtype="step",
                alpha=0.2,
                color="k",
                label="posterior",
            )

            ax[1].hist(
                msini_prior,
                density=True,
                bins=50,
                color="purple",
                alpha=0.2,
                label="prior",
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
            ax[0].legend()
            ax[0].set_xlabel("sma [au]")
            ax[1].set_xlabel("msini [Mjup]")
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
            plt.tight_layout()
            plt.savefig(
                "/home/sblunt/eccentricities/lee_posteriors/{}/priors_{}_pl{}.png".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                dpi=250,
            )
            plt.close()

            rand_samples = np.random.choice(len(sma_posterior), size=1_000)

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
                "/home/sblunt/eccentricities/lee_posteriors/{}/msiniPRIOR_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                msini_prior,
                delimiter=",",
            )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/{}/sma_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_posterior,
                delimiter=",",
            )
            np.savetxt(
                "/home/sblunt/eccentricities/lee_posteriors/{}/smaPRIOR_{}_pl{}.csv".format(
                    savedir, pl[1].hostname, int(pl[1].pl_index)
                ),
                sma_prior,
                delimiter=",",
            )
