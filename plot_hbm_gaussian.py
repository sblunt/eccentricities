
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy import units as u, constants as cst
from scipy.stats import norm

# read in MCMC samples
n_mass_bins = 3
n_sma_bins = 2
n_e_bins = 5
mass_idx = 2
sma_idx = 1

burn_steps = 500  # number of burn-in steps I ran for the actual MCMC
nsteps = 500
nwalkers = 100


ndim = 3
savedir = f"plots/{n_mass_bins}msini{n_sma_bins}sma{n_e_bins}e"

posteriors = np.loadtxt("{}/gaussian_samples_burn{}_total{}_massidx{}_smaidx{}.csv".format(savedir, burn_steps, nsteps, mass_idx, sma_idx),  delimiter=",")
chains = posteriors.reshape((-1, nwalkers, ndim))

"""
plot samples
"""

chains = chains.reshape((-1, ndim))

mu = chains[:,0]
print(np.quantile(mu, [.003, 0.05, 0.16,0.5,0.84,0.95,0.997]))

n2plot = 50
rand_idx = np.random.randint(0, len(chains), size=n2plot)

plt.figure()
for sample in chains[rand_idx]:
    mu, sigma, A = sample

    def ecc_dist(x):
        return A * norm.pdf(x, mu, sigma)
    
    e2plot = np.linspace(0,1, int(1e2))
    plt.plot(e2plot, ecc_dist(e2plot), color='gray', alpha=0.2)
plt.xlabel('ecc.')
plt.ylabel('dN/de dlog(a) dlog(mass)')
plt.savefig(f"{savedir}/gaussian_samples_burn{burn_steps}_total{nsteps}_massidx{mass_idx}_smaidx{sma_idx}.png", dpi=250)

"""
corner plot
"""

corner.corner(chains, labels=['mu','sigma','A'])
plt.savefig(f"{savedir}/gaussian_corner_burn{burn_steps}_total{nsteps}_massidx{mass_idx}_smaidx{sma_idx}.png", dpi=250)

"""
Plot the high-mass vs intermediate-mass sample
"""

posteriors = np.loadtxt("{}/gaussian_samples_burn{}_total{}_massidx2_smaidx{}.csv".format(savedir, burn_steps, nsteps, sma_idx),  delimiter=",")
chains_highmass = posteriors.reshape((-1, ndim))


posteriors = np.loadtxt("{}/gaussian_samples_burn{}_total{}_massidx1_smaidx{}.csv".format(savedir, burn_steps, nsteps, sma_idx),  delimiter=",")
chains_intmass = posteriors.reshape((-1, ndim))
mus_intmass = chains_intmass[:,0]
print(mus_intmass)

fig, ax = plt.subplots()
plt.hist(chains_highmass[:,0], bins=30, histtype='step', density=True, color='k', label='high mass pop.')
plt.hist(chains_intmass[:,0], bins=25, histtype='stepfilled', density=True, color='rebeccapurple', alpha=0.5, label='intermediate mass pop.')
ax.axvline(0.3, color='k', ls='--')
plt.xlim(0,0.8)
plt.xlabel('$\mu$')
plt.legend()
plt.ylabel('relative prob.')
plt.savefig('plots/gaussian_comp.png',dpi=250)