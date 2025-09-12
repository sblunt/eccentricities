"""
a drop-in (faster) replacement for scipy.stats.rv_histogram (not for the general case but for my case; assumes you only need one sample)

NOTE: this doesn't work... WIP
"""

import numpy as np
import matplotlib.pyplot as plt


def rv_histogram(A, xlims):
    """
    assume xlims is [0,x1,x2,...xn,1]
    """

    # compute normalization constant
    C = np.sum((xlims[1:] - xlims[:-1]) * A)

    C = 1 / C

    # print(C)

    u = np.random.uniform(0, 1, size=1)

    # most draws I'm dealing with for this case will be in lowest bin, so we can save time by not calculating the
    # full distribution unless we have to
    for i in np.arange(len(xlims) - 1) + 1:

        cdf_bound = np.sum((xlims[1 : i + 1] - xlims[:i]) * A[:i]) * C
        print(i, cdf_bound)

        # if u < cdf_bound:
        #     subtr_const = 0
        #     for j in range(i):
        #         subtr_const += np.sum((xlims[j] - xlims[j - 1]) * A[j]) * C
        #     return (u - subtr_const) / (C * A[i - 1]) + xlims[i]


if __name__ == "__main__":
    occurrence_rates = np.array([10, 5, 2.5, 1])
    boundaries = np.array([0, 0.3, 0.6, 0.9, 1])

    n_samples = 50_000
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        samples[i] = rv_histogram(occurrence_rates, boundaries)

    plt.figure()
    print(plt.hist(samples, bins=50, density=True))
    plt.savefig("plots/sanity_checks/rv_histogram.png", dpi=250)

    # occurrence_rates = np.array([10, 5])
    # boundaries = np.array([0.3])

    # n_samples = 5_000
    # samples = np.zeros(n_samples)
    # for i in range(n_samples):
    #     samples[i] = rv_histogram(occurrence_rates, boundaries)

    # plt.figure()
    # plt.hist(samples, bins=50, density=True)
    # plt.savefig("plots/sanity_checks/rv_histogram_onebin.png", dpi=250)
