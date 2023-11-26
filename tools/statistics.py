import numpy as np
from tools.config import CI_APPROXIMATION_DISTRIBUTION, CI


class Statistics:

    @staticmethod
    def estimate_expected_value_ci(data, alpha=0.05, method='norm') -> CI:
        """
        Estimate the expected value of the data with a confidence interval.

        Parameters:
        data (array-like): The data to be estimated.
        confidence (float): The confidence level of the interval.

        Returns:
        tuple: The expected value and the confidence interval.
        """
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)

        if method == 'student':
            distribution = CI_APPROXIMATION_DISTRIBUTION.get(method)(n - 1)
        elif method == 'norm':
            distribution = CI_APPROXIMATION_DISTRIBUTION.get(method)
        else:
            raise ValueError(f'Unknown distribution {method} Use one of {CI_APPROXIMATION_DISTRIBUTION.keys()}')

        z = distribution.ppf(1 - alpha / 2)
        ci = z * std / np.sqrt(n)
        return CI(mean - ci, mean + ci)

