import pandas as pd
from radvel.basis import Basis
from radvel.utils import initialize_posterior

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
    print(pl[0])
    if pl[0] == 323:
        starname = pl[1].hostname
        plnum = int(pl[1].pl_index)
        chains = pd.read_csv(
            "/home/sblunt/CLSI/radvel_setup_files/{}/{}_chains.csv.bz2".format(
                starname, starname
            ),
            compression="bz2",
        )
        _, posterior = initialize_posterior(
            "/home/sblunt/CLSI/radvel_setup_files/{}.py".format(starname)
        )
        basis_name = posterior.likelihood.params.basis.name
        n_planets = posterior.likelihood.params.num_planets
        myBasis = Basis(basis_name, n_planets)
        df_synth = myBasis.to_synth(chains)
        ecc_posterior = df_synth["e{}".format(plnum)].values
        print(ecc_posterior)

# TODO: if Lee applied non-uniform ecc priors, I may need to rerun these fits. Check this.
