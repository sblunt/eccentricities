import matplotlib.pyplot as plt
import numpy as np

"""
Fulton+ 23 reproduction plot
Compare with https://content.cld.iop.org/journals/0067-0049/255/1/14/revision1/apjsabfcc1f5_hr.jpg
"""
n_mass_bins = 2
n_e_bins = 1
n_sma_bins=6

nwalkers = 100
ndim = n_mass_bins * n_sma_bins * n_e_bins

chains_msini = np.loadtxt('/home/sblunt/eccentricities/plots/2msini6sma1e_msini/epop_samples_burn500_total500.csv', delimiter=",")
chains_msini = chains_msini.reshape((-1, n_e_bins, n_sma_bins, n_mass_bins))

chains_mass = np.loadtxt('/home/sblunt/eccentricities/plots/2msini6sma1e/epop_samples_burn500_total500.csv', delimiter=",")
chains_mass = chains_mass.reshape((-1, n_e_bins, n_sma_bins, n_mass_bins))



ecc_bins = np.load("completeness_model/{}ecc_bins.npy".format(n_e_bins))
sma_bins = np.load("completeness_model/{}sma_bins.npy".format(n_sma_bins))
msini_bins = np.load("completeness_model/{}msini_bins.npy".format(n_mass_bins))

d_logmsini = np.log(msini_bins[1:]) - np.log(msini_bins[:-1])
d_loga = np.log(sma_bins[1:]) - np.log(sma_bins[:-1])
d_ecc = ecc_bins[1:] - ecc_bins[:-1]

nstars_cps = 719  # total number of stars in the sample

fig, ax = plt.subplots( 1,2, figsize=(10, 5))

# integrate over eccentricity and msini to get dN/d(lna)
n2plot = 100

linewidths = [5,1]
alphas=[0.2,1]

for chain_idx, chains in enumerate([chains_msini, chains_mass]):
    idx2plot = np.random.choice(
        np.arange(len(chains[:, 0, 0, 0])),
        n2plot,
    )
    colors = ["k", "rebeccapurple"]
    fmts = ["o", "^"]
    for i in range(n_mass_bins):

        dn_dmsini_de_dloga = chains[:, :, :, i]  # (n_steps, n_e, n_sma)

        dn_de_dloga = dn_dmsini_de_dloga * d_logmsini[i]
        d_occurrence_de_dloga = dn_de_dloga / nstars_cps * 100

        d_occurrence_dloga = np.sum(d_occurrence_de_dloga * d_ecc, axis=1)

        hist = []

        for j, a in enumerate(sma_bins[:-1]):

            if chain_idx == 0:
                mass_label = "M$\sin{i}$"
                title='no incl. marginalization'
            else:
                mass_label = "M"
                title='with incl. marginalization'

            label = None
            if j == 0 and i == 0:
                label = "{} M$_{{\\oplus}}$ < {} < {} M$_{{\\oplus}}$".format(
                    int(msini_bins[0]), mass_label, int(msini_bins[1])
                )
            elif j == 0 and i == 1:
                label = "{} M$_{{\\oplus}}$ < {} < {} M$_{{\\oplus}}$".format(
                    int(msini_bins[1]), mass_label, int(msini_bins[2])
                )

            # for k in range(n2plot):
            #     ax[chain_idx].plot(
            #         [sma_bins[j], sma_bins[j + 1]],
            #         np.ones(2) * d_occurrence_dloga[idx2plot[k], j],
            #         color=colors[i],
            #         alpha=0.1,
            #     )

            quantiles = np.quantile(d_occurrence_dloga[:, j], [0.16, 0.5, 0.84])
            ax[chain_idx].errorbar(
                [np.exp(0.5 * (np.log(sma_bins[j]) + np.log(sma_bins[j + 1])))],
                [quantiles[1]],
                [[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]],
                color=colors[i],
                fmt=fmts[i],
                label=label,elinewidth=linewidths[i], alpha=alphas[i]
            )

    ax[chain_idx].set_title(title)
    ax[chain_idx].set_xscale("log")
    ax[chain_idx].set_xlabel("$a$ [au]")
    ax[chain_idx].set_ylim(0, 14)
    ax[chain_idx].legend()

ax[0].set_ylabel("N$_{{\\mathrm{{pl}}}}$ / 100 stars / $\\Delta$log$_e$(a)")

plt.savefig(f"plots/fulton_comp.png", dpi=250)
