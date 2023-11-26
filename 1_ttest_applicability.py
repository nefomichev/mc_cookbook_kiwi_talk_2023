from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from tools.vizualization import Visualiser
from tools.config import LOG_NORMAL_EXAMPLE, SAMPLE_SIZE, MONTE_CARLO_ITERS, EXPECTED_VALUE
from tools.statistics import Statistics
from tools.distributions import DistributionGenerator
from scipy.stats import shapiro

"""

Validating statistical test applicability best practices:

1. Measuring FPR and Power is better than validating assumptions separately. 
Assumption validation can be trickier and might have loose thresholds.

2. Simulate AA tests with known effect size and check if the test can estimate the true difference. 
It's better than running AA tests against 0 difference since it's more realistic and can reveal more mistakes.

3. Try different distributions, sample sizes, and effect sizes. Model your real data, 
but also try to break it with corner cases.

5. Rule of thumb: FPR > alpha means underestimation of statistic's variance, issues with core assumptions.
FPR < alpha means underpowered test, the CI is too wide, the test is too conservative.

"""


# Generate the population distribution
Visualiser.visualise_distribution_pdf(LOG_NORMAL_EXAMPLE, title='Population Distribution: Log-normal')

# Monte Carlo simulation
sample_mean_distribution = list()
fpr_count = 0
alpha = 0.05
uplift = 1.5
TRUE_DIFF = (EXPECTED_VALUE * uplift) - EXPECTED_VALUE

mean_difference_distribution = list()
for _ in tqdm(range(MONTE_CARLO_ITERS)):

    # Generate a sample from the population distribution
    control = DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, SAMPLE_SIZE)
    treatment = DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, SAMPLE_SIZE) * uplift
    diff, diff_ci = Statistics.mean_difference_ci(control, treatment, alpha=alpha)
    mean_difference_distribution.append(diff)

    if diff_ci.lower > TRUE_DIFF or diff_ci.upper < TRUE_DIFF:
        fpr_count += 1


# Visualize the sample mean distribution
Visualiser.visualize_sample_distribution(
    mean_difference_distribution,
    percentiles=(2.5, 50, 97.5),
    title=f'Mean difference distribution, Monte Carlo Simulation, Sample size: {SAMPLE_SIZE}, True difference: {TRUE_DIFF}'
)

# Calculate the false positive rate
fpr = fpr_count / MONTE_CARLO_ITERS
fpr_ci_low, fpr_ci_up = proportion_confint(fpr_count, MONTE_CARLO_ITERS, alpha=0.05, method='wilson')
Visualiser.visualise_ci(
    fpr, fpr_ci_low, fpr_ci_up, title=f'False Positive Rate for E[T] - E[C] Confidence Interval, alpha={alpha}'
)
print(f'False positive rate: {fpr:.4f} [{fpr_ci_low:.4f}, {fpr_ci_up:.4f}]')


# Shapiro-Wilk test is a bad choice for MC checks because it is very sensitive to sample size
stat, p_value = shapiro(mean_difference_distribution)

# Output the results
print('P-Value:', p_value)

if p_value > alpha:
    print('Shapiro-Wilk test passed (fail to reject H0)')
else:
    print('ALERT! Shapiro-Wilk test failed (reject H0), t-test could not be applied')
