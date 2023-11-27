from tools.vizualization import Visualiser
from tools.config import LOG_NORMAL_EXAMPLE
from tools.distributions import DistributionGenerator

# Example with tossing a coin
Visualiser.save_plot(
    Visualiser.visualize_freq_probability_view(),
    "freq_probability_view"
)

# Generate the population distribution
Visualiser.save_plot(
    Visualiser.visualise_distribution_pdf(
        LOG_NORMAL_EXAMPLE, title='Population Metric Distribution: Log-normal'
    ),
    "population_model_distribution"
)


# Generate test
Visualiser.save_plot(
    Visualiser.visualize_sample_distribution(
        DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, 1000),
        title='Control Group Sample Distribution: 1000 units'
    ),
    "sample_distribution_control"
)

# Generate control
Visualiser.save_plot(
    Visualiser.visualize_sample_distribution(
        DistributionGenerator.sample_from_distribution(LOG_NORMAL_EXAMPLE, 1000),
        title='Test Group Sample Distribution: 1000 units'
    ),
    "sample_distribution_test"
)
