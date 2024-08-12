import pandas as pd
from radvel.basis import Basis
import pickle
import numpy as np

"""
Grabs ecc, K, and per posteriors for the planet sample and writes them as csvs 
to be ingested into epop!
"""

legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0
)
legacy_planets_inseprange = legacy_planets[
    (legacy_planets.axis > 5) & (legacy_planets.axis < 100)
]
legacy_planets = legacy_planets_inseprange[
    (legacy_planets.mass > 2) & (legacy_planets.mass < 15)
]

# construct the eccentricity posterior for each

for pl in legacy_planets.iterrows():
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
        K_posterior = df_synth["k{}".format(plnum)].values
        per_posterior = df_synth["per{}".format(plnum)].values
        print(np.median(K_posterior))
        print(np.median(per_posterior))

        np.savetxt(
            "/home/sblunt/eccentricities/lee_posteriors/ecc_{}_pl{}.csv".format(
                pl[1].hostname, int(pl[1].pl_index)
            ),
            ecc_posterior,
            delimiter=",",
        )
        np.savetxt(
            "/home/sblunt/eccentricities/lee_posteriors/K_{}_pl{}.csv".format(
                pl[1].hostname, int(pl[1].pl_index)
            ),
            K_posterior,
            delimiter=",",
        )
        np.savetxt(
            "/home/sblunt/eccentricities/lee_posteriors/per_{}_pl{}.csv".format(
                pl[1].hostname, int(pl[1].pl_index)
            ),
            per_posterior,
            delimiter=",",
        )
