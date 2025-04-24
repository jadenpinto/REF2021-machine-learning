import pytest
import pandas as pd
# from pandas.testing import assert_frame_equal
import importlib.util
import sys
from pathlib import Path

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "journal_metrics" / "03_process_source_normalized_impact_per_paper.py"
# Rename module so it does not start with a number
module_name = "proces_source_normalized_impact_per_paper"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
process_snip_df = module.process_snip_df


def test_process_snip_df():
    """
    Function to test how the SNIP DF normalises the ISSNs by splitting print and electron ISSN as their own records
    """
    # Metadata of four journals, two of which were published in 2021:
    # Journal 1: Print ISSN = 000-0002, Electronic ISSN = 000-0006
    # Journal 2: Print ISSN = 000-0003, does not have an electornic ISSN
    raw_snip_df = pd.DataFrame(
        {
            'Year': [2019, 2021, 2021, 2020],
            'Print ISSN': ['000-0001', '000-0002', '000-0003', '000-0004'],
            'Electronic ISSN': ['000-0005', '000-0006', None, '000-0008'],
            'SNIP': [1.684, 1.5, 1.0, 1.33],
            'Another metric': [20, 25, 30, 40]
        }
    )

    actual_processed_snip_df = process_snip_df(raw_snip_df)

    # Filter for the journal's SNIP value published in 2021
    # Normalised such that one row contains on ISSN
    # Drop any nulls i.e. if a journal doesn't have a print or electronic ISSN
    # Drop all the other metrics aside from SNIP
    expected_data = {
        'SNIP': [1.5, 1.0, 1.5],
        'ISSN': ['000-0002', '000-0003', '000-0006']
    }
    expected_processed_snip_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(
        actual_processed_snip_df.reset_index(drop=True),
        expected_processed_snip_df.reset_index(drop=True)
    )


def test_process_snip_df_no_matching_year():
    """
    # Test the SNIP ETL when no rows match the year filter i.e. there is no SNIP value published in 2021
    """
    raw_snip_df = pd.DataFrame({
        'Year': [2019, 2018, 2018, 2022],
            'Print ISSN': ['000-0001', '000-0002', '000-0003', '000-0004'],
            'Electronic ISSN': ['000-0005', '000-0006', None, '000-0008'],
            'SNIP': [1.684, 1.5, 1.0, 1.33],
            'Another metric': [20, 25, 30, 40]
    })

    actual_processed_snip_df = process_snip_df(raw_snip_df)

    # Return empty dataframe
    assert len(actual_processed_snip_df) == 0


def test_process_snip_df_null_issns():
    """
    Test how the SNIP dataframe is processed when all ISSNs are null
    """
    raw_snip_df = pd.DataFrame({
        'Year': [2019, 2021, 2021, 2020],
            'Print ISSN': [None, None, None, None],
            'Electronic ISSN': [None, None, None, None],
            'SNIP': [1.684, 1.5, 1.0, 1.33],
            'Another metric': [20, 25, 30, 40]
    })

    actual_processed_snip_df = process_snip_df(raw_snip_df)

    # Returns empty dataframe:
    assert len(actual_processed_snip_df) == 0
