from unittest.mock import patch, MagicMock
import pandas as pd
import importlib.util
import sys
from pathlib import Path
import pytest

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "output_metrics" / "01_scopus_citation_overview_api.py"
# Rename module so it does not start with a number
module_name = "scopus_citation_overview_api"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
get_citation_metadata = module.get_citation_metadata
extract_citation_metadata = module.extract_citation_metadata
process_citation_metadata = module.process_citation_metadata

# Set up dummy API key
module.elsevier_api_key = "ABCDE1234"

# Test set-up, create resources like dataframes and JSON hashmaps that will be used in multiple tests

@pytest.fixture
def sample_doi_df():
    """Provides a sample DataFrame containing a single DOI used for multiple tests"""
    return pd.DataFrame({
        'DOI': ['10.1145/3034786.3056106']
    })

@pytest.fixture
def sample_citation_metadata_api_payload():
    """Provides a sample JSON payload response to the citations API for the sample DOI values (10.1145/3034786.3056106)"""
    return {
        'abstract-citations-response': {
            'identifier-legend': {'identifier': [{'@_fa': 'true', 'scopus_id': '85021243138'}]},
            'citeInfoMatrix': {
                'citeInfoMatrixXML': {'citationMatrix': {
                    'cc': [{'$': '0'}, {'$': '0'}, {'$': '0'}, {'$': '1'}, {'$': '5'}, {'$': '4'}, {'$': '2'}],
                    'rangeCount': '12'}}},
            'citeColumnTotalXML': {'citeCountHeader': None}}
    }

@pytest.fixture
def sample_extracted_citation_metrics():
    """Provides a sample parsed and extracted JSON containing the citation metrics from the API call for the 10.1145/3034786.3056106 DOI"""
    return {
        'scopus_id': '85021243138', 'citation_counts_2014': 0,
        'citation_counts_2015': 0, 'citation_counts_2016': 0,
        'citation_counts_2017': 1, 'citation_counts_2018': 5,
        'citation_counts_2019': 4, 'citation_counts_2020': 2, 'total_citations': 12
    }


def test_extract_citation_metadata(sample_citation_metadata_api_payload, sample_extracted_citation_metrics):
    expected_extracted_citation_metrics = sample_extracted_citation_metrics

    actual_extracted_citation_metrics = extract_citation_metadata(sample_citation_metadata_api_payload)

    assert isinstance(actual_extracted_citation_metrics, dict)
    assert expected_extracted_citation_metrics == actual_extracted_citation_metrics

def test_extract_empty_citation_metrics():
    assert extract_citation_metadata({}) == {}

def test_extract_citation_metadata_with_none():
    assert extract_citation_metadata(None) == {}

def test_process_citation_metadata(sample_doi_df, sample_citation_metadata_api_payload, sample_extracted_citation_metrics):
    # Expected final dataframe
    expected_df = pd.DataFrame([
        {
            'scopus_id': '85021243138', 'citation_counts_2014': 0,
            'citation_counts_2015': 0, 'citation_counts_2016': 0,
            'citation_counts_2017': 1, 'citation_counts_2018': 5,
            'citation_counts_2019': 4, 'citation_counts_2020': 2,
            'total_citations': 12, 'DOI': '10.1145/3034786.3056106'
        }
    ])

    # Setup mocks
    with patch.object(module, "get_cs_doi_df", return_value=sample_doi_df) as mock_get_cs_doi_df, \
            patch.object(module, "get_citation_metadata", return_value=sample_citation_metadata_api_payload) as mock_get_citation_metadata, \
            patch.object(module, "extract_citation_metadata", return_value=sample_extracted_citation_metrics) as mock_extract_citation_metadata:

        # Process citation metadata for the given DOI:
        result_df = module.process_citation_metadata()

    # Assertions:

    # Assert function calls:
    mock_get_cs_doi_df.assert_called_once()
    mock_get_citation_metadata.assert_called_once()
    mock_extract_citation_metadata.assert_called_once()

    # Assert expected dataframe matches the actual returned dataframe
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_get_citation_metadata(sample_citation_metadata_api_payload):
    sample_doi = "10.1145/3034786.3056106"

    expected_citation_base_url = f"https://api.elsevier.com/content/abstract/citations"
    expected_citation_params = {
        "apiKey": module.elsevier_api_key,
        "doi": sample_doi,
        "httpAccept": "application/json",
        "sort": "+sort-year",
        "date": "2014-2020",
        "field": "scopus_id,cc,rangeCount"
    }

    # Mock the response:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_citation_metadata_api_payload
    mock_response.raise_for_status.return_value = None

    # Test the function with mocked requests.get
    with patch("requests.get", return_value=mock_response) as mock_get:
        result = module.get_citation_metadata(sample_doi)

    # Assertions:

    # Assert API call to the citations API was made - Check requests.get was called
    mock_get.assert_called_once()

    # Parse and assert the call arguments on the mock call
    args, kwargs = mock_get.call_args
    assert args[0] == expected_citation_base_url
    assert kwargs["params"] == expected_citation_params
    assert kwargs["timeout"] == 10

    # Assert that the API responded i.e. mock response was called
    assert mock_response.raise_for_status.called
    assert mock_response.json.called

    # Assert that the citation API returns the mocked JSON payload as expected
    assert result == sample_citation_metadata_api_payload
    assert "abstract-citations-response" in result
    assert "citeInfoMatrix" in result["abstract-citations-response"]
