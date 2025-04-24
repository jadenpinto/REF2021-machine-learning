import pandas as pd
import numpy as np
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "journal_metrics" / "05_handle_missing_journal_metrics.py"
# Rename module so it does not start with a number
module_name = "handle_missing_journal_metrics"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
handle_null_sjr = module.handle_null_sjr


def test_handle_null_sjr():
    """
    Function to test how null SJRs are handled
    """
    # Dataframe containing various journal metrics, some of which have their SJR set to None
    journal_metrics_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': 1.596, 'SJR': None, 'Cite_Score': None},
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.123, 'Cite_Score': 4.001},
        {'ISSN': '0000-0001', 'Scopus_ID': '67890', 'SNIP': 3.001, 'SJR': None, 'Cite_Score': 5.002}
    ])

    # Split journal_metrics into two dataframes, based on if sjr field if None
    null_sjr_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': 1.596, 'SJR': None, 'Cite_Score': None},
        {'ISSN': '0000-0001', 'Scopus_ID': '67890', 'SNIP': 3.001, 'SJR': None, 'Cite_Score': 5.002}
    ])

    not_null_sjr_df = pd.DataFrame([
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.123, 'Cite_Score': 4.001}
    ])

    # Dataframe containing SJR metrics
    sjr_df = pd.DataFrame({
        'Rank': [1, 2],
        'Title': ['Ca-A Cancer Journal for Clinicians', 'Cell'],
        'Issn': ['0000-0001', '1542-4863'],
        'SJR': [56.204, 53.204]
    })

    # Note how the record with ISSN = 0000-0001, with initial null SJR value, now has sjr=56.204, obtained from the sjr_df
    expected_journal_metrics_handled_null_sjr_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': 1.596, 'SJR': None, 'Cite_Score': None},
        {'ISSN': '0000-0001', 'Scopus_ID': '67890', 'SNIP': 3.001, 'SJR': 56.204, 'Cite_Score': 5.002},
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.123, 'Cite_Score': 4.001}
    ])

    with patch.object(module, "split_df_on_null_field", return_value=(null_sjr_df, not_null_sjr_df)) as mock_split_df_on_null_field:
        with patch.object(module, "load_sjr_df", return_value=sjr_df) as mock_load_sjr_df:
            actual_journal_metrics_handled_null_sjr_df = handle_null_sjr(journal_metrics_df)

            # Assert all functions in the ETL were called, with the correct parameters
            mock_split_df_on_null_field.assert_called_once_with(journal_metrics_df, 'SJR')
            mock_load_sjr_df.assert_called_once()

            pd.testing.assert_frame_equal(
                actual_journal_metrics_handled_null_sjr_df.reset_index(drop=True),
                expected_journal_metrics_handled_null_sjr_df.reset_index(drop=True)
            )
