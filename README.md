Welcome to the collection of code I wrote to perform the analysis in Blunt et al (2026): Evidence for a Peak at ~0.3 in the Eccentricity Distribution of Typical Super-Jovian Exoplanets. Please feel free to raise an issue if you spot a bug or have a question.

Here's a quick map for those looking to recreate the analysis:

1. I'm starting from the assumption that you have access to the California Legacy individual fit posteriors (if you don't, reach out to BJ Fulton and he can share with you).
2. Once you have the posteriors (I put them in a directory called `lee_posteriors/run_final`), run `get_posteriors.py`. This script 
grabs eccentricity, msini, and semimajor axis posteriors from `lee_posteriors/run_final` for the sample in my paper, uses importance
resampling to obtain samples the posteriors assuming they were sampled under unifom priors on log(sma) and log(msini), and writes them as csvs to be injested into the HBM model.
3. Next, run `make_completeness_model.py`, which computes a completeness model using publicly available injection-recovery tests (https://github.com/leerosenthalj/CLSI/tree/master/completeness/recoveries_all) 
4. Finally, you can run some hierarchical Bayesian models (HBM). `run_histogram_full_marginalization.py` uses the "accepted" model, which performs a full marginalization over inclination (see appendix A of paper). `run_histogram.py` performs a "cheat-y" marginalization-- instead of doing all the marginalization math, it assumes each msini posterior can be translated to a mass posterior by multiplying by random values of inclination drawn from a uniform distribution in cosi. This script can also be adapted to perform the fit in bins of msini, not mass, by simply setting each value drawn from cosi to be inc=90, rather than the random values. `run_gaussian.py` uses the same "cheat-y" marginalization as `run_histogram.py`, but the HBM model fit is a Gaussian, rather than a histogram. 
5. There are several plotting scripts in the `plotting_scripts` directory that will display the results.

NOTE: I changed around some of the directories right before pushing and didn't recheck that everything works in the new organization. Forgive me, I busy. Please feel free to raise an issue if there's a bug you can't figure out.

NOTE: There's some repeated code in `run_gaussian.py`, `run_histogram_full_marginalziation.py`, and `run_histogram.py`. Forgive me software engineering gods (and future me who will inevitably run into bugs for this reason). 