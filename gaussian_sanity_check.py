from run_epop_gaussian import HierGaussian
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import corner

# NOTE: To run this sanity check, I manually overwrote the completeness to be 1 everywhere in run_epop_gaussian.

# create a sample with some # of posteriors in eccentricity/msini/sma

n_planets = 25

ecc = np.random.normal(0.5,0.001,size=50)
msini = np.random.normal(2000, 1, size=50)
sma = np.random.normal(1,0.01, size=50)

n_msini_bins = 3
n_sma_bins = 2
n_e_bins = 4
mass_idx = 2
sma_idx = 1

like = HierGaussian(
    [ecc]*n_planets,
    msini_posteriors=[msini]*n_planets,
    sma_posteriors=[sma]*n_planets,
    n_sma_bins=n_sma_bins,
    n_e_bins=n_e_bins,
    n_msini_bins=n_msini_bins,
    mass_bin_idx=mass_idx,
    sma_bin_idx=sma_idx
)

print("Running MCMC!")
burn_steps = 500
nwalkers = 100
nsteps = 500

# hbm_samples = like.sample(
#     nsteps,
#     burn_steps=burn_steps,
#     nwalkers=nwalkers,
# )

savedir = f"plots/{n_msini_bins}msini{n_sma_bins}sma{n_e_bins}e"

if not os.path.exists(savedir):
    os.mkdir(savedir)

# np.savetxt(
#     "plots/sanity_checks/sanity_check_samples.csv",
#     hbm_samples,
#     delimiter=",",
# )
chains=np.loadtxt("plots/sanity_checks/sanity_check_samples.csv",delimiter=',')

chains = chains.reshape((-1, 3))

n2plot = 5
rand_idx = np.random.randint(0, len(chains), size=n2plot)

plt.figure()
corner.corner(chains)
# for sample in chains[rand_idx]:
#     mu, sigma, A = sample
#     print(mu, sigma, A)

#     def ecc_dist(x):
#         return A * norm.pdf(x, mu, sigma)
    
#     e2plot = np.linspace(0.4,0.6, int(1e3))
#     plt.plot(e2plot, ecc_dist(e2plot), color='gray', alpha=0.5)
plt.savefig("plots/sanity_checks/sanity_check_gaussian_samples.png", dpi=250)

