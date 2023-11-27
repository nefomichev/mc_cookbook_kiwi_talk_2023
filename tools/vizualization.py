import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tools.config import IMAGE_FOLDER_PATH

FIGSIZE = (10, 6)
STYLE = 'whitegrid'

class Visualiser:
    @staticmethod
    def visualize_sample_distribution(data, percentiles=(25, 50, 75),
                                      title='Data Distribution with Percentile Markers') -> plt:
        """
        Visualize the distribution of the data using Seaborn,
        marking specified percentiles.

        Parameters:
        data (array-like): The data to be visualized.
        percentiles (list): A list of percentiles to mark on the plot.
        """
        plt.style.use('seaborn-v0_8-talk')
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(data, kde=True, ax=ax)

        # Calculate and mark the percentiles
        Visualiser.add_percentiles_plot(data, percentiles, ax)

        # Add legend and labels
        ax.legend(title='Percentiles')
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        plt.show()

        return fig

    @staticmethod
    def add_percentiles_plot(data, percentiles, ax):
        """
        Add a plot of the percentiles of the data to the current plot.
        """
        # Calculate and mark the percentiles
        color_palette = Visualiser.generate_color_palette(len(percentiles))
        for percentile, color in zip(percentiles, color_palette):
            value = np.percentile(data, percentile)
            ax.axvline(x=value, linestyle='--', label=f'{percentile}th percentile: {value:.2f}', color=color)

    @staticmethod
    def generate_color_palette(n_colors):
        return sns.color_palette("hls", n_colors=n_colors)

    @staticmethod
    def visualise_distribution_pdf(distribution, title='PDF'):
        """
        Visualize the probability density function of the distribution
        """
        plt.style.use('seaborn-v0_8-talk')
        fig, ax = plt.subplots(figsize=FIGSIZE)

        x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 1000)

        mean, var, _, _ = distribution.stats(moments='mvsk')
        ax.axvline(x=mean, linestyle='--', label=f'E[X]: {mean:.1f}', color='red')
        ax.axvline(x=mean + np.sqrt(var), linestyle='--', label=f'√D[X]: {np.sqrt(var):.1f}', color='orange')
        ax.plot(x, distribution.pdf(x))

        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()

        plt.show()
        return fig

    @staticmethod
    def visualise_ci(point, ci_lower, ci_upper, title='Single Data Point with Confidence Interval'):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        error = [[point - ci_lower], [ci_upper - point]]
        ax.errorbar(x=[0], y=[point], yerr=error, fmt='o', capsize=5,
                    label=f'{point:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        plt.show()
        return fig

    @staticmethod
    def visualize_freq_probability_view():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.set(style=STYLE)
        title = 'Frequency vs Probability View'

        N = int(1e3)
        coin_flips = np.random.choice([0, 1], size=N)
        cumulative_means = np.cumsum(coin_flips) / (np.arange(N) + 1)
        standard_errors = np.sqrt(cumulative_means * (1 - cumulative_means) / (np.arange(N) + 1))
        ci_upper = cumulative_means + 1.96 * standard_errors
        ci_lower = cumulative_means - 1.96 * standard_errors
        ax.plot(cumulative_means, label='Mean Proportion of Heads', color='blue')
        ax.fill_between(range(N), ci_lower, ci_upper, color='orange', alpha=0.2, label='95% CI')
        ax.axhline(y=.5, linestyle='--', color='red', label='True Heads Probability')

        ax.set_title(title)
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Mean Proportion of Heads')
        ax.set_xlim(0, N)
        ax.set_ylim(0, 1)
        ax.legend()

        plt.show()

        return fig

    @staticmethod
    def save_plot(figure, name):
        figure.savefig(f'{IMAGE_FOLDER_PATH}/{name}.png', dpi=300, bbox_inches='tight')
