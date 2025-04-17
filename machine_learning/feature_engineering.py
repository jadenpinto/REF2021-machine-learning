import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata
from utils.constants import FIGURES_DIR


def main():
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    for col in cs_outputs_enriched_metadata.columns:
        unique_val_count = cs_outputs_enriched_metadata[col].nunique()
        print(f"Field {col} has {unique_val_count} unique values")

    print(cs_outputs_enriched_metadata.dtypes)

    check_skewness_total_citations(cs_outputs_enriched_metadata)

    a = transform_and_normalise_citations(cs_outputs_enriched_metadata)
    print(a['normalised_citations'].describe())

def check_skewness_total_citations(cs_outputs_enriched_metadata):
    """
    Function to test if the total citations feature is skewed
    :param cs_outputs_enriched_metadata: Dataframe containing the total_citations field
    :return: None
    """
    # Drop all nulls from total citations
    total_citations_col = cs_outputs_enriched_metadata['total_citations'].dropna()
    total_citations_col_skew = skew(total_citations_col)

    # Plot histogram:
    plt.figure(figsize=(10, 6)) # Bigger fig size than default, if not used, plot looks too zoomed out
    plt.hist(total_citations_col, bins=300)
    plt.title(f'Distribution of Total Citations \nSkewness = {total_citations_col_skew}')
    plt.xlabel('Total Citations')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save plot:
    total_citations_skew_test_path = os.path.join(
        os.path.dirname(__file__), "..", FIGURES_DIR, "total_citations_skew_test.png"
    )
    plt.savefig(total_citations_skew_test_path)

    plt.show()

    # Check if the feature is skewed based on its skew value
    # Interpreting results of scipy skew metric:
    if total_citations_col_skew > 1:
        print(f"total_citations is highly positively skewed - Right-skewed (tail on the right).")
    elif total_citations_col_skew < -1:
        print(f"total_citations is highly negatively skewed - Left-skewed (tail on the left)")
    elif -0.5 < total_citations_col_skew < 0.5:
        print(f"total_citations is symmetrical.") # Near zero skew implies Symmetric distribution
    else:
        print(f"total_citations is skewed (moderate).")

def transform_and_normalise_citations(df):
    # Create a copy - avoid modifying the original dataframe
    result_df = df.copy()

    # Apply natural log transformation - use log1p which treats ln(0) as ln(1) = 0, since ln(0) is undefined
    result_df['normalised_citations'] = np.log1p(result_df['total_citations'])

    year_groups = result_df.groupby('Year') # Group by year to normalise within each year

    # For each year, calculate mean and std-dev of log-transformed citations
    year_stats = year_groups['normalised_citations'].agg(['mean', 'std']).reset_index()

    result_df = pd.merge(result_df, year_stats, on='Year', how='left') # Merge year_stats back to the main dataframe

    # Normalise by year: z-score normalisation: Handling edge cases where std-dev might be 0 for a year [Standard Scalar]
    result_df['normalised_citations'] = result_df.apply(
        lambda row: (row['normalised_citations'] - row['mean']) / row['std'] if row['std'] > 0 else 0,
        axis=1
    )

    result_df = result_df.drop(['mean', 'std'], axis=1) # Drop the temporary columns

    return result_df

def infer_missing_top_citation_percentile(cs_outputs_enriched_metadata):
    # infer_top_percentiles: make Nulls to 100% since it's implied
    cs_outputs_enriched_metadata = cs_outputs_enriched_metadata.copy()
    cs_outputs_enriched_metadata[['top_citation_percentile']] = cs_outputs_enriched_metadata[['top_citation_percentile']].fillna(value=100.0)
    return cs_outputs_enriched_metadata

def log_transform_author_count(cs_outputs_enriched_metadata):
    df = cs_outputs_enriched_metadata.copy()
    # Apply log transformation - using log1p function which treats ln(0) as ln(1) = 0 since ln(0) is undefined

    df['log_transformed_authors'] = np.log1p(df['Number of additional authors'].fillna(0))
    return df

def get_cs_outputs_df(input_set):
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()

    if "citations" in input_set:
        cs_outputs_enriched_metadata = transform_and_normalise_citations(cs_outputs_enriched_metadata)

    if "scival output metrics" in input_set:
        cs_outputs_enriched_metadata = infer_missing_top_citation_percentile(cs_outputs_enriched_metadata)

    if "author metrics" in input_set:
        cs_outputs_enriched_metadata = log_transform_author_count(cs_outputs_enriched_metadata)

    return cs_outputs_enriched_metadata

def get_cluster_features(input_set):
    features = []

    if "citations" in input_set:
        features.append('normalised_citations')

    if "journal metrics" in input_set:
        journal_impact_metrics = ['SNIP', 'SJR', 'Cite_Score']
        features.extend(journal_impact_metrics)

    if "scival output metrics" in input_set:
        output_metrics = ['field_weighted_citation_impact', 'top_citation_percentile', 'field_weighted_views_impact']
        features.extend(output_metrics)

    if "author metrics" in input_set:
        author_metrics = ['log_transformed_authors']
        features.extend(author_metrics)

    return features


if __name__ == "__main__":
    main()


"""
One hot encoding:

Field Institution UKPRN code has 90 unique values
Field Institution name has 90 unique values
Field Output type has 13 unique values
Field Title has 7005 unique values
Field Place has 43 unique values
Field Publisher has 97 unique values
Field Year has 8 unique values
Field Number of additional authors has 40 unique values
Field Interdisciplinary has 1 unique values
Field Forensic science has 1 unique values
Field Criminology has 1 unique values
Field Research group has 102 unique values
Field Open access status has 8 unique values
Field Cross-referral requested has 17 unique values
Field Delayed by COVID19 has 1 unique values
Field Incl sig material before 2014 has 2 unique values
Field Incl reseach process has 2 unique values
Field Incl factual info about significance has 2 unique values
"""

# todo first look at metrics, possibly move it into it's own file.
# Next, look at feature engineering, consider different options and scaling too.
