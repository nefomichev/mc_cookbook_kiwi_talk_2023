from tools.distributions import DistributionGenerator
from tools.helper import Helper
from scipy.stats import norm, t
from collections import namedtuple

IMAGE_FOLDER_PATH = Helper.find_root_path() / 'images'

# Configuration
EXPECTED_VALUE = 10
VARIANCE = 49
SAMPLE_SIZE = int(1e3)
MONTE_CARLO_ITERS = int(1e4)
ALPHA = 0.05

# Generate the population distribution
LOG_NORMAL_EXAMPLE = DistributionGenerator.generate_lognormal(EXPECTED_VALUE, VARIANCE)


# Structs
CI = namedtuple('ci', ['lower', 'upper'])

CI_APPROXIMATION_DISTRIBUTION = {
    'norm': norm,
    'student': t
}
