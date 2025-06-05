import pandas as pd
import importlib.util
import sys
from pathlib import Path

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "journal_metrics" / "04_process_scimago_journal_rank.py"
# Rename module so it does not start with a number
module_name = "process_scimago_journal_rank"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
filter_sjr_columns = module.filter_sjr_columns
log_issn_lengths = module.log_issn_lengths
drop_null_issn = module.drop_null_issn
normalise_to_1nf = module.normalise_to_1nf
add_hyphen_issn = module.add_hyphen_issn
handle_sjr_issn = module.handle_sjr_issn


def test_filter_sjr_columns():
    """
    Test to check function that selects a list of specified columns from the SJR dataset
    """
    original_sjr_df = pd.DataFrame({
        'Rank': [1, 2, 3],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology', 'Cell'],
        'Issn': ['15424863, 00079235', '14710072, 14710080', '00928674, 10974172'],
        'SJR': [56.204, 33.213, 25.716],
        'H index': ['211', '508', '892'],
        'Total Docs. (2021)': [38, 108, 513]
    })

    actual_filtered_sjr_df = filter_sjr_columns(original_sjr_df)

    expected_filtered_sjr_df = pd.DataFrame({
        'Rank': [1, 2, 3],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology', 'Cell'],
        'Issn': ['15424863, 00079235', '14710072, 14710080', '00928674, 10974172'],
        'SJR': [56.204, 33.213, 25.716]
    })

    # Assert the data is filtered to retain only the specified columns
    pd.testing.assert_frame_equal(actual_filtered_sjr_df, expected_filtered_sjr_df)
    assert list(actual_filtered_sjr_df.columns) == ['Rank', 'Title', 'Issn', 'SJR']


def test_drop_null_issn():
    """
    Test the function that drops records where the ISSN is '-'
    """
    original_sjr_df = pd.DataFrame({
        'Rank': [1, 2, 3],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology', 'Journal3'],
        'Issn': ['15424863, 00079235', '14710072, 14710080', '-'],
        'SJR': [56.204, 33.213, 25.716]
    })

    actual_sjr_df_with_no_blank_issns = drop_null_issn(original_sjr_df)

    # The third journal is dropped as its Issn equals '-'
    expected_sjr_df_with_no_blank_issns = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology'],
        'Issn': ['15424863, 00079235', '14710072, 14710080'],
        'SJR': [56.204, 33.213]
    })

    pd.testing.assert_frame_equal(actual_sjr_df_with_no_blank_issns, expected_sjr_df_with_no_blank_issns)


def test_add_hyphen_issn():
    """
    Function to test method that splits ISSNs into two parts and adds a hyphen in between
    """
    original_sjr_df = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology'],
        'Issn': ['15424863', '14710072'],
        'SJR': [56.204, 33.213]
    })

    actual_sjr_df_transformed_issn = add_hyphen_issn(original_sjr_df)

    expected_sjr_df_transformed_issn = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Nature Reviews Molecular Cell Biology'],
        'Issn': ['1542-4863', '1471-0072'],
        'SJR': [56.204, 33.213]
    })

    pd.testing.assert_frame_equal(actual_sjr_df_transformed_issn, expected_sjr_df_transformed_issn)

    # Double check by iterating through all issns
    for issn in actual_sjr_df_transformed_issn['Issn']:
        assert issn[4] == '-'
        assert len(issn) == 9

def test_normalise_to_1nf():
    """
    Function that tests method responsible for normalising ISSNs such that one record has one ISSN
    """
    original_sjr_df = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Cell'],
        'Issn': ['15424863, 00079235', '14710072, 14710080'],
        'SJR': [56.204, 33.213]
    })

    actual_normalised_issn_sjr_df = normalise_to_1nf(original_sjr_df)

    # Normalise DF: split records containing multiple ISSNs such that there is one ISSN per row
    expected_normalised_issn_sjr_df = pd.DataFrame({
        'Rank': [1, 1, 2, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Ca-A Cancer Journal for Clinicians', 'Cell', 'Cell'],
        'Issn': ['15424863', '00079235', '14710072', '14710080'],
        'SJR': [56.204, 56.204, 33.213, 33.213]
    })

    pd.testing.assert_frame_equal(
        actual_normalised_issn_sjr_df.reset_index(drop=True),
        expected_normalised_issn_sjr_df.reset_index(drop=True)
    )

def test_handle_sjr_issn():
    """
    Function to test how original ISSN dataframes are transformed
    """
    original_sjr_df = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Cell'],
        'Issn': ['15424863, 00079235', '-'],
        'SJR': [56.204, 33.213]
    })

    actual_transformed_sjr_df = handle_sjr_issn(original_sjr_df)

    # Transformation: Drop '-' ISSNs, normalise ISSNs (one per row), split ISSN and add hyphen
    expected_transformed_sjr_df = pd.DataFrame({
        'Rank': [1, 1],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Ca-A Cancer Journal for Clinicians'],
        'Issn': ['1542-4863', '0007-9235'],
        'SJR': [56.204, 56.204]
    })

    pd.testing.assert_frame_equal(
        actual_transformed_sjr_df.reset_index(drop=True),
        expected_transformed_sjr_df.reset_index(drop=True)
    )


def test_log_issn_lengths(capfd):
    """
    Function to test logger method that prints the different lengths of ISSNs in the SJR dataset
    """
    sjr_df = pd.DataFrame({
        'Rank': [1, 2, 3, 4],
        'Title': ['Journal 1', 'Journal 2', 'Journal 3', 'Journal 4'],
        'Issn': ['15424863, 00079235', '00079234', '-', '14710072, 14710080'],
        'SJR': [56.204, 33.213, 45.322, 22.334]
    })

    # Call logger function, to log the different lengths of ISSNs
    log_issn_lengths(sjr_df)

    # Capture standard output
    out, err = capfd.readouterr()

    # Assert that the correct output was printed - the log sorts the lengths first
    assert out == "Possible Lengths of ISSNs in SJR dataframe=[np.int64(1), np.int64(8), np.int64(18)]\n"
