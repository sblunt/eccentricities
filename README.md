Welcome to the collection of code I wrote to perform the analysis in Blunt et al (2026): Evidence for a Peak at ~0.3 in the Eccentricity Distribution of Typical Super-Jovian Exoplanets. Please feel free to raise an issue if you spot a bug or have a question.

Here's a quick map for those looking to recreate the analysis:

1. I'm starting from the assumption that you have access to the California Legacy individual fit posteriors (if you don't, reach out to BJ Fulton and he can share with you).
2. Once you have the posteriors (I put them in a directory called `lee_posteriors/run_final`), run `get_posteriors.py`. This script 
grabs eccentricity, msini, and semimajor axis posteriors from `lee_posteriors/run_final` for the sample in my paper, uses importance
resampling to obtain samples the posteriors assuming they were sampled under unifom priors on log(sma) and log(msini), and writes them as csvs to be injested into the HBM model.
3. Next, run `make_completeness_model.py`, which computes a completeness model using publicly available injection-recovery tests (https://github.com/leerosenthalj/CLSI/tree/master/completeness/recoveries_all) 