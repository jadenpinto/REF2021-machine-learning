import pytest
import pandas as pd
from unittest.mock import patch

from machine_learning.clustering_one import infer_cluster_labels

@pytest.fixture
def cluster_training_data():
    """
    Set-up a cluster training dataframe containing data points representing outputs containing metadata about
    which university published it, and which cluster they are assigned to
    """
    return pd.DataFrame({
        'Institution UKPRN code': [10007783, 10007783, 10007856, 10007856, 10000001, 10000001],
        'cluster': [0, 0, 1, 1, 0, 1]
    })

@pytest.fixture
def enhanced_results_df():
    """
    Set up the enhanced results dataframe where each university has a count for the number of high and low scoring outputs
    """
    return pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856, 10000001],
        'high_scoring_outputs': [3, 0, 5],
        'low_scoring_outputs': [0, 2, 0]
    })

@patch('machine_learning.clustering_one.get_high_scoring_universities')
def test_infer_cluster_labels(mock_get_high_scoring_universities, cluster_training_data, enhanced_results_df):
    #  The high scoring universities are: 10007783 and 10000001, return them in the function returning high-scoring universities
    mock_get_high_scoring_universities.return_value = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10000001]
    })

    actual_cluster_labels_dict = infer_cluster_labels(cluster_training_data, enhanced_results_df)

    # The cluster assignments of data points belonging to high-scoring university:
    # 10007783: 0, 0
    # 10000001: 0, 1
    # 0 count = 3, 1 count = 1. 0 count > 1 count. Cluster 0 represents high-scoring outputs
    expected_cluster_labels_dict = {
        0: 'high_scoring_outputs',
        1: 'low_scoring_outputs'
    }

    assert actual_cluster_labels_dict == expected_cluster_labels_dict


