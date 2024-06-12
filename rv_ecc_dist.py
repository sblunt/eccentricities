# goal of this file: given
# 1) the ecc posteriors of all RV planets in CLS sample, and
# 2) the completeness contours from completeness.py:
# do the following:
# 1) calculate the pop-level eccentricity distribution for the RV planets in a sma/msini range that overlaps with the imaged planets
# 2) compare the overall pop-level eccentricity distribution for all RV CLS planets to other publications (as a sanity check)

import pandas as pd
import matplotlib.pyplot as plt

# read in the eccentricity posteriors of all RV planets in CLS sample
# note: the chains themselves are not public (as far as I could see), so I
# reconstructed posteriors based on the med/lo/hi statistics

legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0
)
# TODO: pretty sure this is selecting by the mode of the distribution, but check that

# select a subset that overlap in mass and sma with the imaged objects in Brendan's sample
# Brendan's cuts: M between 2 and 75 Mjup, (projected!) seps 5-100 au. We are going by
# separation, not projected separation. BDs: 15-75 Mjup, GPs: 2-15 Mjup
legacy_planets_inseprange = legacy_planets[
    (legacy_planets.axis > 5) & (legacy_planets.axis < 100)
]

# TODO: check if this is mass or msini (I'm assuming mass)
legacy_bds = legacy_planets_inseprange[
    (legacy_planets.mass > 15) & (legacy_planets.mass < 75)
]
legacy_planets = legacy_planets_inseprange[
    (legacy_planets.mass > 2) & (legacy_planets.mass < 15)
]

plt.figure()
plt.scatter(legacy_bds.axis.values, legacy_bds.mass.values, color="purple")
plt.scatter(
    legacy_planets.axis.values, legacy_planets.mass.values, color="white", ec="purple"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("semimajor axis [au]")
plt.ylabel("mass [M$_{{\\mathrm{{jup}}}}$]")
plt.savefig("plots/legacy_sample.png", dpi=250)

plt.figure()
plt.hist(legacy_planets.e_med.values, bins=7, label="Legacy RV planets")
# TODO: overplot imaged planets
plt.legend()
plt.ylabel("N. planets")
plt.xlabel("median eccentricity")
plt.savefig("plots/legacy_ecc_histogram.png", dpi=250)


print("Number of BDs in Legacy sample: {}".format(len(legacy_bds)))
print("Number of planets in Legacy sample: {}".format(len(legacy_planets)))
