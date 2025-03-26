import pandas as pd
import numpy as np

from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata


def main():
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    for col in cs_outputs_enriched_metadata.columns:
        unique_val_count = cs_outputs_enriched_metadata[col].nunique()
        print(f"Field {col} has {unique_val_count} unique values")

    print(cs_outputs_enriched_metadata.dtypes)

    a = transform_and_normalise_citations(cs_outputs_enriched_metadata)
    print(a['normalised_citations'].describe())


def transform_and_normalise_citations(df):
    # Create a copy - avoid modifying the original dataframe
    result_df = df.copy()

    # Apply natural log transformation - adding 1 to prevent issues with computing ln(0) which is undefined
    result_df['normalised_citations'] = np.log(result_df['total_citations'] + 1)

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

def get_cs_outputs_df(input_set):
    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()

    if "citations" in input_set:
        cs_outputs_enriched_metadata = transform_and_normalise_citations(cs_outputs_enriched_metadata)

    return cs_outputs_enriched_metadata

def get_cluster_features(input_set):
    features = []

    if "citations" in input_set:
        features.append('normalised_citations')

    if "journal metrics" in input_set:
        # journal_impact_metrics = ['SNIP', 'SJR', 'Cite_Score']
        journal_impact_metrics = ['SNIP']
        features.extend(journal_impact_metrics)

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
