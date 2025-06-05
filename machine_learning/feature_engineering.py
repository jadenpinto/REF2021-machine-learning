import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata
from utils.constants import FIGURES_DIR


def main():
    """
    Performs feature engineering to transform features prior to use in the clustering model.
    Includes skewness correction, temporal standardisation, and inferring missing values.
    """
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    for col in cs_outputs_enriched_metadata.columns:
        unique_val_count = cs_outputs_enriched_metadata[col].nunique()
        print(f"Field {col} has {unique_val_count} unique values")

    print(cs_outputs_enriched_metadata.dtypes)

    check_skewness_total_citations(cs_outputs_enriched_metadata)

    cs_outputs_enriched_metadata_with_transformed_citations = transform_and_normalise_citations(cs_outputs_enriched_metadata)
    print(cs_outputs_enriched_metadata_with_transformed_citations['normalised_citations'].describe())

def check_skewness_total_citations(cs_outputs_enriched_metadata):
    """
    Function to test if the total citations feature is skewed
    :param cs_outputs_enriched_metadata: Dataframe containing enriched metadata of CS outputs including total citations
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
    """
    Apply log transformation on total citations to correct skew, and normalise by year for i.e. Temporal Standardisation
    :param df: Dataframe containing enriched metadata of CS outputs including total citations
    :return: Dataframe containing enriched metadata of CS outputs with the total citations log transformed and year normalised
    """
    # Create a copy - avoid modifying the original dataframe
    result_df = df.copy()

    # Apply natural log transformation - use log1p which treats ln(0) as ln(1) = 0, since ln(0) is undefined
    result_df['normalised_citations'] = np.log1p(result_df['total_citations'])

    year_groups = result_df.groupby('Year') # Group by year to normalise within each year

    # For each year, calculate mean and std-dev of log-transformed citations
    year_stats = year_groups['normalised_citations'].agg(['mean', 'std']).reset_index()

    result_df = pd.merge(result_df, year_stats, on='Year', how='left') # Merge year_stats back to the main dataframe

    # Normalise by year: z-score normalisation: Handle edge case where std-dev might be 0 for a year [Standard Scalar]
    result_df['normalised_citations'] = result_df.apply(
        lambda row: (row['normalised_citations'] - row['mean']) / row['std'] if row['std'] > 0 else 0,
        axis=1
    )

    result_df = result_df.drop(['mean', 'std'], axis=1) # Drop the temporary mean and std columns

    return result_df

def infer_missing_top_citation_percentile(cs_outputs_enriched_metadata):
    """
    If an output is missing value for top citation percentile it means the output was not in the top 1, 5, 10, 25, or 50
    percent of highly cited papers, therefore it is within the 100th percentile, and are hence null is replaced with 100.
    :param cs_outputs_enriched_metadata: Dataframe containing enriched metadata of CS outputs including top citation percentile
    :return: Dataframe containing enriched metadata of CS outputs where missing top citation percentiles are inferred
    """
    # infer_top_percentiles: make Nulls to 100% since it's implied
    cs_outputs_enriched_metadata = cs_outputs_enriched_metadata.copy()
    cs_outputs_enriched_metadata[['top_citation_percentile']] = cs_outputs_enriched_metadata[['top_citation_percentile']].fillna(value=100.0)
    return cs_outputs_enriched_metadata

def log_transform_author_count(cs_outputs_enriched_metadata):
    """
    The number of additional authors is a skewed feature - apply log transformation for Skewness Correction
    :param cs_outputs_enriched_metadata: Dataframe containing enriched metadata of CS outputs including number of
    additional authors
    :return: Dataframe containing enriched metadata of CS outputs with the number of additional authors log transformed
    to handle skew
    """
    df = cs_outputs_enriched_metadata.copy()
    # Apply log transformation - using log1p function which treats ln(0) as ln(1) = 0 since ln(0) is undefined

    df['log_transformed_authors'] = np.log1p(df['Number of additional authors'].fillna(0))
    return df

def get_cs_outputs_df(features):
    """
    Return the CS outputs enriched metadata with features engineered
    :param features: Features used to train the clustering models
    :return: CS outputs enriched metadata with features engineered
    """
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()

    if "normalised_citations" in features:
        # Log transform, Year normalise
        cs_outputs_enriched_metadata = transform_and_normalise_citations(cs_outputs_enriched_metadata)

    if "top_citation_percentile" in features:
        # Replace nulls with 100
        cs_outputs_enriched_metadata = infer_missing_top_citation_percentile(cs_outputs_enriched_metadata)

    if "log_transformed_authors" in features:
        # Log transform
        cs_outputs_enriched_metadata = log_transform_author_count(cs_outputs_enriched_metadata)

    return cs_outputs_enriched_metadata


if __name__ == "__main__":
    main()
