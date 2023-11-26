import numpy as np
from scipy.stats import lognorm


class DistributionGenerator:

    @staticmethod
    def generate_lognormal(expected_value, variance):
        """
        Generate log-normal distributed data with specified mean and variance.

        Parameters:
        mean (float): Desired mean of the log-normal distribution.
        variance (float): Desired variance of the log-normal distribution.
        size (int): Number of samples to generate.

        Returns:
        array: Log-normal distributed data.
        """
        sigma_squared = np.log(1 + variance / expected_value ** 2)
        mu = np.log(expected_value) - sigma_squared / 2
        return lognorm(s=np.sqrt(sigma_squared), scale=np.exp(mu))

    @staticmethod
    def sample_from_distribution(distribution, sample_size):
        return distribution.rvs(sample_size)
