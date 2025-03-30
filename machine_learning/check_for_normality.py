import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata
from machine_learning.feature_engineering import infer_missing_top_citation_percentile


def main():
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    cs_outputs_enriched_metadata = infer_missing_top_citation_percentile(cs_outputs_enriched_metadata)
    # print(cs_outputs_enriched_metadata.head().to_string())
    features = [
        'Number of additional authors', 'SNIP', 'SJR', 'Cite_Score', 'total_citations',
        'field_weighted_citation_impact', 'top_citation_percentile', 'field_weighted_views_impact'
    ]
    plot_histograms(cs_outputs_enriched_metadata, features)
    # plot_qq(cs_outputs_enriched_metadata, features)

def plot_histograms(cs_outputs_enriched_metadata, features):
    for feature in features:
        plt.figure(figsize=(10, 5))

        # Handle missing values - Drop all nulls
        data = cs_outputs_enriched_metadata[feature].dropna()

        # Plot histogram
        plt.hist(data, bins='auto', density=True,alpha=0.6, color='g', label='Data')

        # Overlay normal distribution
        mu = data.mean()
        sigma = data.std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        pdf_fmt = 'r-' # Red line to represent the normal fit

        plt.plot(x, pdf, pdf_fmt, linewidth=2, label='Normal Fit')

        plt.title(f'Distribution of {feature} - μ={mu:.2f}, σ={sigma:.2f}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_qq(cs_outputs_enriched_metadata, features):
    # Quantile-Quantile Plot

    for feature in features:
        # https://stackoverflow.com/questions/13865596/quantile-quantile-plot-using-scipy
        return


if __name__ == "__main__":
    main()


"""
1. quantile-quantile (QQ) plots - recommended 
2. You can look at a histogram of the data, does the shape look similar to a normal distribution? 
3. You can do a hypothesis test to formally test this (Shapiro-Wilk test, etc)



Try increasing the number of bins in your histogram plot. You can also try visualising your data with a qq plot,
looking at other statistics such as kurtosis or perform test for normality.
"""