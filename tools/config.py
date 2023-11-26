from tools.distributions import DistributionGenerator
from scipy.stats import norm, t
from collections import namedtuple

# Configuration
EXPECTED_VALUE = 10
VARIANCE = 50
SAMPLE_SIZE = int(1e3)
MONTE_CARLO_ITERS = int(1e4)

# Generate the population distribution
LOG_NORMAL_EXAMPLE = DistributionGenerator.generate_lognormal(EXPECTED_VALUE, VARIANCE)


# Structs
CI = namedtuple('ci', ['lower', 'upper'])

CI_APPROXIMATION_DISTRIBUTION = {
    'norm': norm,
    'student': t
}
