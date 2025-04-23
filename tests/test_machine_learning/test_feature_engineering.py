import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from machine_learning.feature_engineering import infer_missing_top_citation_percentile, transform_and_normalise_citations, log_transform_author_count

def test_infer_missing_top_citation_percentile():
    # CS output metadata containing the top citation percentiles field
    cs_outputs_enriched_metadata = pd.DataFrame({
        'top_citation_percentile': [25.0, None, 50.0, None]
    })

    # CS outputs metadata with the null top citation percentiles replaced with 100
    # Since the output is not in the top 1, 5, 15, 50 percentile, it can be said to be in the top 100% of percentiles
    expected_cs_outputs_enriched_metadata_updated = pd.DataFrame({
        'top_citation_percentile': [25.0, 100.0, 50.0, 100.0]
    })

    actual_cs_outputs_enriched_metadata_updated = infer_missing_top_citation_percentile(cs_outputs_enriched_metadata)

    pd.testing.assert_frame_equal(
        actual_cs_outputs_enriched_metadata_updated,
        expected_cs_outputs_enriched_metadata_updated
    )


def test_transform_and_normalise_citations():
    """
    Test the function that log transforms and normalises citations by publication year
    """

    # CS metadata consisting of Year of publication and the total citations
    cs_metadata_df = pd.DataFrame(
        {
            'Year': [2015, 2015, 2016, 2017, 2019, 2019, 2019, 2019],
            'total_citations': [0, 1, 10, 0, 0, 200, 100, 1000]
        }
    )

    # Obtain expected dataframe with transformed citations (by log transformation + normalising by publication year)
    expected_cs_metadata_df_with_transformed_citations = cs_metadata_df.copy()
    expected_cs_metadata_df_with_transformed_citations['normalised_citations_log'] = np.log1p(expected_cs_metadata_df_with_transformed_citations['total_citations'])

    year_stats_expected = expected_cs_metadata_df_with_transformed_citations.groupby('Year')['normalised_citations_log'].agg(['mean', 'std']).reset_index()

    expected_cs_metadata_df_with_transformed_citations = pd.merge(expected_cs_metadata_df_with_transformed_citations, year_stats_expected, on='Year', how='left')

    expected_cs_metadata_df_with_transformed_citations['normalised_citations'] = expected_cs_metadata_df_with_transformed_citations.apply(
        lambda row: (row['normalised_citations_log'] - row['mean']) / row['std'] if row['std'] > 0 else 0,
        axis=1
    )

    # Match column order to the output format
    expected_cs_metadata_df_with_transformed_citations = expected_cs_metadata_df_with_transformed_citations[['Year', 'total_citations', 'normalised_citations']]

    actual_cs_metadata_df_with_transformed_citations = transform_and_normalise_citations(cs_metadata_df)

    pd.testing.assert_frame_equal(
        actual_cs_metadata_df_with_transformed_citations,
        expected_cs_metadata_df_with_transformed_citations
    )


def test_log_transform_author_count():
    """
    Function to test log transformation on the number of additional authors
    """
    additional_author_of_outputs_list = [0, 0, 12, 10, 1]

    # CS metadata consisting of number of additional authors of outputs
    cs_outputs_enriched_metadata = pd.DataFrame({
        'Number of additional authors': additional_author_of_outputs_list
    })

    # Log transform the number of additional authors, and add to dataframe in a new column, log_transformed_authors
    expected_cs_outputs_enriched_metadata_log_transformed = cs_outputs_enriched_metadata.copy()
    expected_log_transformed = np.log1p(additional_author_of_outputs_list)
    expected_cs_outputs_enriched_metadata_log_transformed['log_transformed_authors'] = expected_log_transformed

    actual_cs_outputs_enriched_metadata_log_transformed = log_transform_author_count(cs_outputs_enriched_metadata)

    pd.testing.assert_frame_equal(
        actual_cs_outputs_enriched_metadata_log_transformed,
        expected_cs_outputs_enriched_metadata_log_transformed
    )
