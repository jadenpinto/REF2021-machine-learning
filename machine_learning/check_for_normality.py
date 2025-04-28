import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot

from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata
from machine_learning.feature_engineering import infer_missing_top_citation_percentile
from utils.constants import FIGURES_DIR


def main():
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    cs_outputs_enriched_metadata = infer_missing_top_citation_percentile(cs_outputs_enriched_metadata)
    # print(cs_outputs_enriched_metadata.head().to_string())
    features = [
        'Number of additional authors', 'SNIP', 'SJR', 'Cite_Score', 'total_citations',
        'field_weighted_citation_impact', 'top_citation_percentile', 'field_weighted_views_impact'
    ]
    plot_histograms(cs_outputs_enriched_metadata, features)
    plot_qq(cs_outputs_enriched_metadata, features)
    statistical_normality_tests(cs_outputs_enriched_metadata, features)

def plot_histograms(cs_outputs_enriched_metadata, features):
    for feature in features:
        plt.figure(figsize=(10, 5))

        # Handle missing values - Drop all nulls
        data = cs_outputs_enriched_metadata[feature].dropna()

        # Plot histogram
        plt.hist(data, bins='auto', density=True,alpha=0.6, color='b', label='Data')

        # Overlay normal distribution
        mean = data.mean()
        std_dev = data.std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = stats.norm.pdf(x, mean, std_dev)
        pdf_fmt = 'r-' # Red line to represent the normal fit

        plt.plot(x, pdf, pdf_fmt, linewidth=2, label='Normal Fit')

        plt.title(f'Distribution of {feature} - μ={mean:.2f}, σ={std_dev:.2f}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        histogram_for_normality_check_path = os.path.join(
            os.path.dirname(__file__), "..", FIGURES_DIR, feature + '_histogram_normality_check'
        )
        plt.savefig(histogram_for_normality_check_path)

        plt.show()


def plot_qq(cs_outputs_enriched_metadata, features):
    # Quantile-Quantile Plot

    for feature in features:
        data = cs_outputs_enriched_metadata[feature].dropna()

        line_type = 's' # standardised line - scaled by the std dev of the feature with mean added
        qqplot(data, line=line_type)

        plt.title(f'Quantile-Quantile Plot - {feature}')
        plt.grid(True)
        plt.tight_layout()

        qq_plot_for_normality_check_path = os.path.join(
            os.path.dirname(__file__), "..", FIGURES_DIR, feature + '_qq_plot_normality_check'
        )
        plt.savefig(qq_plot_for_normality_check_path)

        plt.show()

def statistical_normality_tests(cs_outputs_enriched_metadata, features):
    # Assumption: Sample is drawn from a Gaussian distribution (Data is normal) => Null hypothesis / H0
    # Threshold: Used to interpret the p value => Alpha, here: 5% = 0.05
    # Assumption holds true (sample likely drawn from a Gaussian distribution) if p value is greater than threshold

    # p <= alpha: reject H0, not normal.
    # p > alpha: fail to reject H0, normal.
    alpha = 0.05

    # 1) Shapiro-Wilk Test {usually suitable for smaller samples of data}
    print("Shapiro-Wilk Test:")
    for feature in features:
        data = cs_outputs_enriched_metadata[feature].dropna()
        stat, p = stats.shapiro(data)
        print(f"Statistics = {stat:.2f}, p = {p:.2f}", end= " ==> ")
        if p > alpha:
            print(f'{feature} is normal') # Fail to reject H0
        else:
            print(f'{feature} is not normal') # Reject H0

    # Per Shapiro - none of my features are normal, all have p=0.
    # UserWarning: scipy.stats.shapiro: For N > 5000, computed p-value may not be accurate. (All of features have N>5000)
    print()

    # 2) D'Agostino's K^2 Test
    print("D'Agostino's K^2 Test:")
    for feature in features:
        data = cs_outputs_enriched_metadata[feature].dropna()
        stat, p = stats.normaltest(data)
        print(f"Statistics = {stat:.2f}, p = {p:.2f}", end=" ==> ")
        if p > alpha:
            print(f'{feature} is normal')  # Fail to reject H0
        else:
            print(f'{feature} is not normal')  # Reject H0

    # Again, none of my features are normal, all have p=0
    print()

    # 3) Anderson-Darling Test
    print("Anderson-Darling Test:")
    for feature in features:
        data = cs_outputs_enriched_metadata[feature].dropna()
        anderson_result = stats.anderson(data)
        print(f"{feature}: Statistic = {anderson_result.statistic :.2f}")

        for i in range(len(anderson_result.critical_values)):
            sl, cv = anderson_result.significance_level[i], anderson_result.critical_values[i]
            if anderson_result.statistic < anderson_result.critical_values[i]:
                print(f'Significance Level of {sl:.2f}: Critical Value = {cv:.2f}, data is normal')  # Fail to reject H0
            else:
                print(f'Significance Level of {sl:.2f}: Critical Value = {cv:.2f}, data is not normal')  # Reject H0

if __name__ == "__main__":
    main()
