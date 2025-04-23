import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from machine_learning.cs_output_results import enhance_score_distribution, log_high_low_scoring_universities, get_high_scoring_universities, log_university_count, filter_ref_outputs_for_cs

@pytest.fixture
def input_data_frames():
    """
    Input dataframes function returning:
        1. CS output results dataframe [Results of each university - distributed across output level quality scores]
        2. CS Outputs enriched metadata [Includes CS outputs with complete metadata]
    """
    # Specify results dataframe, with 2 universities
    # Uni 1, 10007783. Distribution: 50%, 20%, 30%, 0%, 0%
    # Uni 2, 10007856. Distribution: 40%, 40%, 10%, 10%, 0%
    cs_output_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856],
        '4*': [50, 40],
        '3*': [20, 40],
        '2*': [30, 10],
        '1*': [0, 10],
        'Unclassified': [0, 0]
    })

    # Specify the outputs dataframe, with 2 universities
    # University with UKPRN 10007783 has 10 outputs
    # University with UKPRN 10007856 also has 10 outputs
    cs_outputs_enriched_metadata = pd.DataFrame({
        'Institution UKPRN code': [
            10007783, 10007783, 10007783, 10007783, 10007783, 10007783, 10007783, 10007783, 10007783, 10007783,
            10007856, 10007856, 10007856, 10007856, 10007856, 10007856, 10007856, 10007856, 10007856, 10007856
        ]
    })

    return cs_output_results_df, cs_outputs_enriched_metadata

def test_enhance_score_distribution(input_data_frames):
    # Obtained dataframes for results and outputs
    cs_output_results_df, cs_outputs_enriched_metadata = input_data_frames

    # Enhance the df by merging the results with outputs to obtain high/low scoring counts
    actual_enhanced_score_distribution_df = enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata)

    expected_enhanced_score_distribution_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856],
        '4*': [50, 40],
        '3*': [20, 40],
        '2*': [30, 10],
        '1*': [0, 10],
        'Unclassified': [0, 0],
        'total_university_outputs': [10, 10],
        '4star_outputs': [5, 4],
        '3star_outputs': [2, 4],
        '2star_outputs': [3, 1],
        '1star_outputs': [0, 1],
        'unclassified_outputs': [0, 0],
        'high_scoring_outputs': [7, 8],
        'low_scoring_outputs': [3, 2]
    })

    assert_frame_equal(
        actual_enhanced_score_distribution_df.reset_index(drop=True),
        expected_enhanced_score_distribution_df,
        check_dtype=False
    )

def test_log_high_low_scoring_universities(capfd):
    # Two universities 10007783, 10007856 which have 0 low scoring outputs and 7, 8 high scoring ones, respectively
    enhanced_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856],
        'high_scoring_outputs': [7, 8],
        'low_scoring_outputs': [0, 0]
    })

    # Call the function to log number of high and low scoring universities using the enhanced dataframe
    log_high_low_scoring_universities(enhanced_results_df)
    # Capture standard output
    out, err = capfd.readouterr()

    assert "Number of universities where all CS outputs were scored low = 0" in out
    assert "Number of universities where all CS outputs were scored high = 2" in out

def test_get_high_scoring_universities():
    # Dataframe with 5 universities, 2 of which only have high scoring outputs - 10007856, 10007857
    enhanced_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856, 10007857, 10007858, 10007859],
        'high_scoring_outputs': [10, 5, 32, 0, 13],
        'low_scoring_outputs': [2, 0, 0, 13, 15]
    })

    # Obtain DF containing the high scoring universities
    actual_high_scoring_universities_df = get_high_scoring_universities(enhanced_results_df)
    # Sort by their UKPRN
    actual_high_scoring_universities_df_sorted = actual_high_scoring_universities_df.sort_values(
        by='Institution code (UKPRN)').reset_index(drop=True)

    # Expected dataframe of high scoring universities, sorted by UKPRN
    expected_high_scoring_universities_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007856, 10007857]
    }).sort_values(by='Institution code (UKPRN)').reset_index(drop=True)

    assert_frame_equal(actual_high_scoring_universities_df_sorted, expected_high_scoring_universities_df)

def test_log_university_count(capfd):
    # Dataframe of university results, for two universities
    cs_output_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007856, 10007857]
    })

    # Function to log university count
    log_university_count(cs_output_results_df)
    # Capture standard output
    out, err = capfd.readouterr()

    assert "The total number of universities who have submitted CS outputs to REF2021: 2" in out


def test_filter_ref_outputs_for_cs():
    # Results from 4 universities two of which are for the outputs profile (10007856, 10007857)
    cs_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007856, 10000001, 10007857, 10000002],
        'Profile': ['Outputs', 'Impact', 'Outputs', 'Environment'],
    })

    # Results dataframe filtered to retain Outputs result and drop any Impact or Environment results
    expected_cs_output_results_df = pd.DataFrame({
        'Institution code (UKPRN)': [10007856, 10007857],
        'Profile': ['Outputs', 'Outputs']
    }).reset_index(drop=True)

    actual_cs_output_results_df = filter_ref_outputs_for_cs(cs_results_df).reset_index(drop=True)

    assert_frame_equal(actual_cs_output_results_df, expected_cs_output_results_df)
