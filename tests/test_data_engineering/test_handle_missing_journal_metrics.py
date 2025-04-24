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
handle_null_snip = module.handle_null_snip
handle_missing_journal_metrics = module.handle_missing_journal_metrics
ensure_uniform_data_types = module.ensure_uniform_data_types


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

    # Split journal_metrics into two dataframes, based on if sjr field is None
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


def test_handle_null_snip():
    """
    Function to test how null SJRs are handled
    """
    # Dataframe containing various journal metrics, some of which have their SNIPs set to None
    journal_metrics_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': None, 'SJR': 2.999, 'Cite_Score': 3.456},
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.456, 'Cite_Score': 4.001},
        {'ISSN': '2190-572X', 'Scopus_ID': '67890', 'SNIP': None, 'SJR': 1.234, 'Cite_Score': 5.001}
    ])

    # Split journal_metrics into two dataframes, based on if SNIP field is None
    null_snip_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': None, 'SJR': 2.999, 'Cite_Score': 3.456},
        {'ISSN': '2190-572X', 'Scopus_ID': '67890', 'SNIP': None, 'SJR': 1.234, 'Cite_Score': 5.001}
    ])

    not_null_snip_df = pd.DataFrame([
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.456, 'Cite_Score': 4.001}
    ])

    # Dataframe containing SNIP metrics
    snip_df = pd.DataFrame([
        {'SNIP': 0.854926629627807, 'ISSN': '2190-572X'},
        {'SNIP': 1.01023499, 'ISSN': '9999-9991'}
    ])

    # Note how the record with ISSN = 2190-572X, with initial null SNIP value, now has SNIP=0.854926629627807 (from snip_df)
    expected_journal_metrics_handled_null_snip_df = pd.DataFrame([
        {'ISSN': '2168-2305', 'Scopus_ID': None, 'SNIP': None, 'SJR': 2.999, 'Cite_Score': 3.456},
        {'ISSN': '2190-572X', 'Scopus_ID': '67890', 'SNIP': 0.854926629627807, 'SJR': 1.234, 'Cite_Score': 5.001},
        {'ISSN': '1234-5678', 'Scopus_ID': '12345', 'SNIP': 2.123, 'SJR': 3.456, 'Cite_Score': 4.001}
    ])

    with patch.object(module, "split_df_on_null_field",
                      return_value=(null_snip_df, not_null_snip_df)) as mock_split_df_on_null_field:
        with patch.object(module, "load_processed_snip_df", return_value=snip_df) as mock_load_processed_snip_df:

            actual_journal_metrics_handled_null_snip_df = handle_null_snip(journal_metrics_df)

            # Assert all functions in the ETL were called, with the correct parameters
            mock_split_df_on_null_field.assert_called_once_with(journal_metrics_df, 'SNIP')
            mock_load_processed_snip_df.assert_called_once()

            pd.testing.assert_frame_equal(
                actual_journal_metrics_handled_null_snip_df.reset_index(drop=True),
                expected_journal_metrics_handled_null_snip_df.reset_index(drop=True)
            )


def test_handle_missing_journal_metrics():
    """
    Function to test how missing journal metrics are handled - checks to see if the ETL calls the required functions
    """
    # Original journal metrics with missing SNIP and SJR
    journal_metrics_df = pd.DataFrame([
        {'ISSN': '0000-0001', 'Scopus_ID': '00000', 'SNIP': None, 'SJR': None, 'Cite_Score': 3.009}
    ])

    # Updated journal metrics with missing SJR handled
    journal_metrics_df_transformed_sjr = pd.DataFrame([
        {'ISSN': '0000-0001', 'Scopus_ID': '00000', 'SNIP': None, 'SJR': 3.999, 'Cite_Score': 3.009}
    ])

    # Final journal metrics with missing SJR and SNIP handled
    expected_journal_metrics_df_transformed_sjr_snip = pd.DataFrame([
        {'ISSN': '0000-0001', 'Scopus_ID': '00000', 'SNIP': 3.001, 'SJR': 3.999, 'Cite_Score': 3.009}
    ])

    with patch.object(module, "handle_null_sjr", return_value=journal_metrics_df_transformed_sjr) as mock_handle_sjr:
        with patch.object(module, "handle_null_snip", return_value=expected_journal_metrics_df_transformed_sjr_snip) as mock_handle_snip:

            actual_journal_metrics_df_transformed_sjr_snip = handle_missing_journal_metrics(journal_metrics_df)

            # First, the handle missing SJR is called on the original dataframe
            mock_handle_sjr.assert_called_once_with(journal_metrics_df)
            # Next, the handle missing SNIP on the dataframe with the transformed SJR
            mock_handle_snip.assert_called_once_with(journal_metrics_df_transformed_sjr)

            pd.testing.assert_frame_equal(
                actual_journal_metrics_df_transformed_sjr_snip.reset_index(drop=True),
                expected_journal_metrics_df_transformed_sjr_snip.reset_index(drop=True)
            )


def test_ensure_uniform_data_types():
    """
    Function to test the method transformed the values of the journal metrics dataframe to specified data types
    """
    # Journal metrics DF where each column contains several mix of data types like float, int and string
    journal_metrics_df = pd.DataFrame({
        'ISSN': ['1234-5678', '9876-5432'],
        'Scopus_ID': [12345, '67890'],
        'SNIP': [1.234, '2.345'],
        'SJR': ['3.456', 4.567],
        'Cite_Score': [5.678, '6.789']
    })

    # Call the function
    actual_transformed_journal_metrics_df = ensure_uniform_data_types(journal_metrics_df)

    # Assert:

    # Numeric columns are converted to floats
    assert actual_transformed_journal_metrics_df['SNIP'].dtype == np.float64
    assert actual_transformed_journal_metrics_df['SJR'].dtype == np.float64
    assert actual_transformed_journal_metrics_df['Cite_Score'].dtype == np.float64

    # String columns must be of type string (Scopus_ID is not converted to scientific notation)
    assert pd.api.types.is_string_dtype(actual_transformed_journal_metrics_df['ISSN'])
    assert pd.api.types.is_string_dtype(actual_transformed_journal_metrics_df['Scopus_ID'])
    assert actual_transformed_journal_metrics_df.loc[0, 'Scopus_ID'] == '12345'
