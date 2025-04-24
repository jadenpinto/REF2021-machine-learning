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
