import numpy as np
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from tools.vizualization import Visualiser
from tools.config import LOG_NORMAL_EXAMPLE, SAMPLE_SIZE, MONTE_CARLO_ITERS, EXPECTED_VALUE
from tools.statistics import Statistics
from tools.distributions import DistributionGenerator

# Generate the population distribution
Visualiser.visualise_distribution_pdf(LOG_NORMAL_EXAMPLE, title='Population Distribution: Log-normal')

# Monte Carlo simulation
sample_mean_distribution = list()
fpr_count = 0
alpha = 0.05
for _ in tqdm(range(MONTE_CARLO_ITERS)):

    # Generate a sample from the population distribution
    sample = DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, SAMPLE_SIZE)
    sample_mean = np.mean(sample)
    sample_mean_distribution.append(sample_mean)

    # Try to estimate the E[X] using the sample mean
    ci = Statistics.estimate_expected_value_ci(sample, alpha=alpha, method='norm')
    if ci.lower > EXPECTED_VALUE or ci.upper < EXPECTED_VALUE:
        fpr_count += 1

# Visualize the sample mean distribution
Visualiser.visualize_sample_distribution(
    sample_mean_distribution,
    percentiles=(2.5, 50, 97.5),
    title=f'Sample mean distribution, Monte Carlo Simulation, Sample size: {SAMPLE_SIZE}, Iterations: {MONTE_CARLO_ITERS}'
)

# Calculate the false positive rate
fpr = fpr_count / MONTE_CARLO_ITERS
fpr_ci_low, fpr_ci_up = proportion_confint(fpr_count, MONTE_CARLO_ITERS, alpha=0.05, method='wilson')
Visualiser.visualise_ci(
    fpr, fpr_ci_low, fpr_ci_up, title=f'False Positive Rate for E[X] Confidence Interval, alpha={alpha}'
)
print(f'False positive rate: {fpr:.4f} [{fpr_ci_low:.4f}, {fpr_ci_up:.4f}]')
