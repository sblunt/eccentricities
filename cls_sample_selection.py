"""
Select RV sample that overlaps with Brendan's cuts in sma/mass
"""

import pandas as pd
import matplotlib.pyplot as plt

# select a subset from the CLS confirmed planets sample that overlap with Brendan's cuts

legacy_planets = pd.read_csv(
    "/home/sblunt/CLSI/legacy_tables/planet_list.csv", index_col=0
)

# remove false positives
legacy_planets = legacy_planets[
    (legacy_planets.status == "K")
    | (legacy_planets.status == "S")
    | (legacy_planets.status == "C")
]

# select a subset that overlap in mass and sma with the imaged objects in Brendan's sample
# Brendan's cuts: M between 2 and 75 Mjup, (projected!) seps 5-100 au. We are going by
# separation, not projected separation. BDs: 15-75 Mjup, GPs: 2-15 Mjup
legacy_planets_inseprange = legacy_planets[
    (legacy_planets.axis > 5) & (legacy_planets.axis < 100)
]

# this is making a selection by the median model msini = most probable mass
legacy_bds = legacy_planets_inseprange[
    (legacy_planets.mass_med > 15) & (legacy_planets.mass_med < 50)
]

# TODO: make a stronger statistical argument that the samples overlap in mass
# TODO: make a statistical argument that the samples overlap in sep/projected sep

legacy_planets = legacy_planets_inseprange[
    (legacy_planets.mass_med > 2) & (legacy_planets.mass_med < 15)
]

plt.figure()
plt.scatter(legacy_bds.axis.values, legacy_bds.mass_med.values, color="purple")
plt.scatter(
    legacy_planets.axis.values,
    legacy_planets.mass_med.values,
    color="white",
    ec="purple",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("semimajor axis [au]")
plt.ylabel("Msini [M$_{{\\mathrm{{jup}}}}$]")
plt.savefig("plots/legacy_sample.png", dpi=250)

plt.figure()
plt.hist(legacy_planets.e_med.values, bins=7, label="Legacy RV planets")
plt.legend()
plt.ylabel("N. planets")
plt.xlabel("median eccentricity")
plt.savefig("plots/legacy_ecc_histogram.png", dpi=250)

print("Number of BDs in Legacy sample: {}".format(len(legacy_bds)))
print("Number of planets in Legacy sample: {}".format(len(legacy_planets)))

print("Legacy planet host names: {}".format(legacy_planets.hostname))

"""
120066 pl 1
145675 pl 2
156279 pl 2
181234 pl 1
213472 pl 1
217014 pl 2
217107 pl 2
26161 pl 1
28185 pl 2
4203 pl 2
50499 pl 2
66428 pl 2
75732 pl 3
92788 pl 2
183263 pl 2
"""
