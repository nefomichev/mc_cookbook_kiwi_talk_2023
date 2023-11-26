import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


FIGSIZE = (10, 6)
STYLE = 'whitegrid'


class Visualiser:
    @staticmethod
    def visualize_sample_distribution(data, percentiles=(25, 50, 75),
                                      title='Data Distribution with Percentile Markers') -> None:
        """
        Visualize the distribution of the data using Seaborn,
        marking specified percentiles.

        Parameters:
        data (array-like): The data to be visualized.
        percentiles (list): A list of percentiles to mark on the plot.
        """
        sns.set(style=STYLE)
        plt.figure(figsize=FIGSIZE)
        sns.histplot(data, kde=True)

        # Calculate and mark the percentiles
        Visualiser.add_percentiles_plot(data, percentiles)

        # Add legend and labels
        plt.legend(title='Percentiles')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def add_percentiles_plot(data, percentiles):
        """
        Add a plot of the percentiles of the data to the current plot.
        """
        # Calculate and mark the percentiles
        color_palette = Visualiser.generate_color_palette(len(percentiles))
        for percentile, color in zip(percentiles, color_palette):
            value = np.percentile(data, percentile)
            plt.axvline(x=value, linestyle='--', label=f'{percentile}th percentile: {value:.2f}', color=color)

    @staticmethod
    def generate_color_palette(n_colors):
        return sns.color_palette("hls", n_colors=n_colors)

    @staticmethod
    def visualise_distribution_pdf(distribution, title='PDF'):
        """
        Visualize the probability density function of the distribution
        """
        sns.set(style=STYLE)
        plt.figure(figsize=FIGSIZE)

        x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 1000)

        mean, var, _, _ = distribution.stats(moments='mvsk')
        plt.axvline(x=mean, linestyle='--', label=f'E[X]: {mean:.1f}', color='red')
        plt.axvline(x=mean, linestyle='--', label=f'D[X]: {var:.1f}', color='red')
        plt.legend(title='Statistics')
        plt.plot(x, distribution.pdf(x))
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.show()

    @staticmethod
    def visualise_ci(point, ci_lower, ci_upper, title='Single Data Point with Confidence Interval'):
        error = [[point - ci_lower], [ci_upper - point]]

        # Plotting
        sns.set(style=STYLE)
        plt.figure(figsize=FIGSIZE)

        plt.errorbar(x=[0], y=[point], yerr=error,
                     fmt='o', capsize=5, label=f'{point:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]')
        plt.xticks([])  # Hides x-axis ticks
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.show()
