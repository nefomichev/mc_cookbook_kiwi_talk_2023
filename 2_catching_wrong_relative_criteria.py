import numpy as np
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from tools.vizualization import Visualiser
from tools.config import LOG_NORMAL_EXAMPLE, SAMPLE_SIZE, MONTE_CARLO_ITERS, EXPECTED_VALUE
from tools.statistics import Statistics
from tools.distributions import DistributionGenerator


"""
This is an example on how we can catch a wrong criteria. Imagine we want to estimate the relative difference between
the control and treatment groups. 

- Can we use the mean difference confidence interval and divide it by the control mean to get the relative difference?
- No! This approach is wrong. 

It underestimates the variance of the relative difference caused by an extra noise in denominator. MC helps to catch it.

"""

# Generate the population distribution
Visualiser.visualise_distribution_pdf(LOG_NORMAL_EXAMPLE, title='Population Distribution: Log-normal')

# Monte Carlo simulation
alpha = 0.05
uplift = 1.5
TRUE_DIFF = (EXPECTED_VALUE * uplift) - EXPECTED_VALUE
RELATIVE_TRUE_DIFF = TRUE_DIFF / EXPECTED_VALUE

mean_difference_distribution = list()
fpr_count = 0
sample_mean_distribution = list()
for _ in tqdm(range(MONTE_CARLO_ITERS)):

    # Generate a sample from the population distribution
    control = DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, SAMPLE_SIZE)
    treatment = DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, SAMPLE_SIZE) * uplift
    diff, diff_ci = Statistics.mean_difference_ci(control, treatment, alpha=alpha)

    relative_diff = diff / np.mean(control)
    mean_difference_distribution.append(relative_diff)

    # THIS IS WRONG, IT'S NOT A VALID RELATIVE CI ESTIMATION, MONTE CARLO SHOULD SHOW IT
    if diff_ci.lower / np.mean(control) > RELATIVE_TRUE_DIFF or diff_ci.upper / np.mean(control) < RELATIVE_TRUE_DIFF:
        fpr_count += 1

# Visualize the sample mean distribution
Visualiser.visualize_sample_distribution(
    mean_difference_distribution,
    percentiles=(2.5, 50, 97.5),
    title=f'Mean difference distribution, Monte Carlo Simulation, Sample size: {SAMPLE_SIZE}, True difference: {RELATIVE_TRUE_DIFF}'
)

# Calculate the false positive rate
fpr = fpr_count / MONTE_CARLO_ITERS
fpr_ci_low, fpr_ci_up = proportion_confint(fpr_count, MONTE_CARLO_ITERS, alpha=0.05, method='wilson')
Visualiser.visualise_ci(
    fpr, fpr_ci_low, fpr_ci_up, title=f'False Positive Rate for (E[T] - E[C]) / E[C] Confidence Interval, alpha={alpha}'
)
print(f'False positive rate: {fpr:.4f} [{fpr_ci_low:.4f}, {fpr_ci_up:.4f}]')


