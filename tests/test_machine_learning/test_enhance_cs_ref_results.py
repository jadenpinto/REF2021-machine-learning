import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from machine_learning.cs_output_results import enhance_score_distribution, log_high_low_scoring_universities

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
    data = pd.DataFrame({
        'Institution code (UKPRN)': [10007783, 10007856],
        'high_scoring_outputs': [7, 8],
        'low_scoring_outputs': [0, 0]
    })

    # Call the function to log number of high and low scoring universities using the enhanced dataframe
    log_high_low_scoring_universities(data)
    # Capture standard output
    out, err = capfd.readouterr()

    assert "Number of universities where all CS outputs were scored low = 0" in out
    assert "Number of universities where all CS outputs were scored high = 2" in out
